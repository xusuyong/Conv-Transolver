import torch
import torch.nn as nn
from .base_model import BaseModel
from .neighbor_ops import (
    NeighborSearchLayer,
    NeighborMLPConvLayer,
    NeighborMLPConvLayerLinear,
    NeighborMLPConvLayerWeighted,
)

from neuralop.models import FNO
from neuralop.models.tfno import Projection

from .net_utils import PositionalEmbedding, AdaIN, MLP


class GNOFNOGNO(BaseModel):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=64,
        hidden_channels=(86, 86),
        in_channels=1,
        out_channels=1,
        fno_modes=(32, 32, 32),
        fno_hidden_channels=86,
        fno_out_channels=86,
        fno_domain_padding=0.125,
        fno_norm="group_norm",
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
    ):
        super().__init__()
        self.weighted_kernel = weighted_kernel
        self.nb_search_in = NeighborSearchLayer(radius_in)
        self.nb_search_out = NeighborSearchLayer(radius_out)
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.df_embed = MLP([in_channels, embed_dim, 3 * embed_dim], torch.nn.GELU)
        self.linear_kernel = linear_kernel

        kernel1 = MLP([10 * embed_dim, 512, 256, hidden_channels[0]], torch.nn.GELU)
        self.gno1 = NeighborMLPConvLayerWeighted(mlp=kernel1)

        if linear_kernel == False:
            kernel2 = MLP(
                [fno_out_channels + 4 * embed_dim, 512, 256, hidden_channels[1]],
                torch.nn.GELU,
            )
            self.gno2 = NeighborMLPConvLayer(mlp=kernel2)
        else:
            kernel2 = MLP([7 * embed_dim, 512, 256, hidden_channels[1]], torch.nn.GELU)
            self.gno2 = NeighborMLPConvLayerLinear(mlp=kernel2)

        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=hidden_channels[0] + 3 + in_channels,
            out_channels=fno_out_channels,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=fno_norm,
            rank=fno_rank,
        )

        self.projection = Projection(
            in_channels=hidden_channels[1],
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=nn.functional.gelu,
            n_dim=1,
        )

    # x_in : (n_in, 3)
    # x_out : (n_x, n_y, n_z, 3)
    # df : (in_channels, n_x, n_y, n_z)
    # ara : (n_in)

    # u : (n_in, out_channels)
    # @torch.autocast(device_type="cuda")
    def forward(self, x_in, x_out, df, x_eval=None, area_in=None, area_eval=None):
        """
        x_in = data_dict["vertices"].squeeze(0)  # (n_in, 3)
        x_out = (data_dict["sdf_query_points"].squeeze(0).permute(1, 2, 3, 0))  # (n_x, n_y, n_z, 3)
        df = data_dict["sdf"]  # (1, n_x, n_y, n_z)
        """
        # manifold to latent neighborhood
        in_to_out_nb = self.nb_search_in(x_in, x_out.view(-1, 3))

        # latent to manifold neighborhood
        if x_eval is not None:
            out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), x_eval)
        else:
            out_to_in_nb = self.nb_search_out(x_out.view(-1, 3), x_in)

        # Embed manifold coordinates
        resolution = df.shape[-1]  # 16,32,64
        n_in = x_in.shape[0]
        if area_in is None or self.weighted_kernel is False:
            area_in = torch.ones((n_in,), device=self.device)
        x_in = torch.cat([x_in, area_in.unsqueeze(-1)], dim=-1) # ([3586, 4])顶点 和 面积 组合
        x_in_embed = self.pos_embed(
            x_in.reshape(
                -1,
            )
        ).reshape(
            (n_in, -1)
        )  # (n_in, 4*embed_dim)  调用pos_embed() torch.Size([3586, 128])

        if x_eval is not None:
            n_eval = x_eval.shape[0]
            if area_eval is None or self.weighted_kernel is False:
                area_eval = torch.ones((n_eval,), device=self.device)
            x_eval = torch.cat([x_eval, area_eval.unsqueeze(-1)], dim=-1)
            x_eval_embed = self.pos_embed(
                x_eval.reshape(
                    -1,
                )
            ).reshape(
                (n_eval, -1)
            )  # (n_eval, 4*embed_dim)

        # Embed latent space coordinates
        x_out_embed = self.pos_embed(
            x_out.reshape(
                -1,
            )
        ).reshape(
            (resolution**3, -1)
        )  # (n_x*n_y*n_z, 3*embed_dim)  调用pos_embed() torch.Size([262144, 96])

        # Embed latent space features
        df_embed = self.df_embed(df.permute(1, 2, 3, 0)).reshape(
            (resolution**3, -1)
        )  # (n_x*n_y*n_z, 3*embed_dim) 调用MLP torch.Size([262144, 96])
        grid_embed = torch.cat(
            [x_out_embed, df_embed], dim=-1
        )  # (n_x*n_y*n_z, 6*embed_dim)  把这两组合在一起 torch.Size([262144, 192])

        # GNO : project to latent space
        u = self.gno1(
            x_in_embed, in_to_out_nb, grid_embed, area_in
        )  # (n_x*n_y*n_z, hidden_channels[0])   #torch.Size([262144, 64])
        u = (
            u.reshape(resolution, resolution, resolution, -1)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
        )  # (1, hidden_channels[0], n_x, n_y, n_z)  torch.Size([1, 64, 64, 64, 64])

        # Add positional embedding and distance information  在这里把三个组合一起放进FNO
        u = torch.cat(
            (x_out.permute(3, 0, 1, 2).unsqueeze(0), df.unsqueeze(0), u), dim=1
        )  # (1, 3+in_channels+hidden_channels[0], n_x, n_y, n_z)  torch.Size([1, 68, 64, 64, 64])

        # FNO on latent space
        u = self.fno(u)  # (1, fno_out_channels, n_x, n_y, n_z)  torch.Size([1, 64, 64, 64, 64])
        u = (
            u.squeeze().permute(1, 2, 3, 0).reshape(resolution**3, -1)
        )  # (n_x*n_y*n_z, fno_out_channels)  torch.Size([262144, 64])

        # GNO : project to manifold
        if self.linear_kernel == False:
            if x_eval is not None:
                u = self.gno2(
                    u, out_to_in_nb, x_eval_embed
                )  # (n_eval, hidden_channels[1])
            else:
                u = self.gno2(u, out_to_in_nb, x_in_embed)  # (n_in, hidden_channels[1])
        else:
            if x_eval is not None:
                u = self.gno2(
                    x_in=x_out_embed,
                    neighbors=out_to_in_nb,
                    in_features=u,
                    x_out=x_eval_embed,
                )
            else:
                u = self.gno2(
                    x_in=x_out_embed,
                    neighbors=out_to_in_nb,
                    in_features=u,
                    x_out=x_in_embed,
                ) #torch.Size([3586, 64])
        u = u.unsqueeze(0).permute(0, 2, 1)  # (1, hidden_channels[1], n_in/n_eval)  torch.Size([1, 64, 3586])

        # Pointwise projection to out channels
        u = self.projection(u).squeeze(0).permute(1, 0)  # (n_in/n_eval, out_channels)  torch.Size([3586, 1])

        return u  # ([3586, 1])

    # Batch size 1 is assumed

    def data_dict_to_input(self, data_dict):
        x_in = data_dict["vertices"].squeeze(0)  # (n_in, 3)
        x_out = (
            data_dict["sdf_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["sdf"]  # (1, n_x, n_y, n_z)

        x_in, x_out, df = (
            x_in.to(self.device),
            x_out.to(self.device),
            df.to(self.device),
        )

        return x_in, x_out, df

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        pred = self(x_in, x_out, df).reshape(1, -1)
        if loss_fn is None:
            loss_fn = self.loss
        truth = data_dict["pressure"].to(self.device).reshape(1, -1)
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df = self.data_dict_to_input(data_dict)
        pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss
        return {
            "loss": loss_fn(
                pred.view(1, -1), data_dict["pressure"].view(1, -1).to(self.device)
            )
        }


class GNOFNOGNOAhmed(GNOFNOGNO):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=16,
        hidden_channels=(86, 86),
        in_channels=2,
        out_channels=1,
        fno_modes=(32, 32, 32),
        fno_hidden_channels=86,
        fno_out_channels=86,
        fno_domain_padding=0.125,
        fno_norm="ada_in",
        adain_embed_dim=64,
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
        max_in_points=5000,
        subsample_train=1,
        subsample_eval=1,
    ):
        if fno_norm == "ada_in":
            init_norm = "group_norm"
        else:
            init_norm = fno_norm

        self.max_in_points = max_in_points
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval

        super().__init__(
            radius_in=radius_in,
            radius_out=radius_out,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_out_channels=fno_out_channels,
            fno_domain_padding=fno_domain_padding,
            fno_norm=init_norm,
            fno_factorization=fno_factorization,
            fno_rank=fno_rank,
            linear_kernel=linear_kernel,
            weighted_kernel=weighted_kernel,
        )

        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = nn.ModuleList(
                AdaIN(adain_embed_dim, fno_hidden_channels)
                for _ in range(
                    self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers
                )
            )
            self.use_adain = True
        else:
            self.use_adain = False

    def data_dict_to_input(self, data_dict):
        x_in = data_dict["centroids"][0]  # (n_in, 3)
        x_out = (
            data_dict["df_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["df"]  # (1, n_x, n_y, n_z)
        area = data_dict["areas"][0]  # (n_in, 3)

        # info_fields = torch.cat([
        #    v*torch.ones_like(df) for _, v in data_dict['info'][0].items()
        # ], dim=0) # (8, n_x, n_y, n_z)

        info_fields = data_dict["info"][0]["velocity"] * torch.ones_like(df)

        df = torch.cat((df, info_fields), dim=0)  # (2, n_x, n_y, n_z)  这里解释了in_chanels=2

        if self.use_adain:
            vel = (
                torch.tensor([data_dict["info"][0]["velocity"]])
                .view(
                    -1,
                )
                .to(self.device)
            )
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)

        x_in, x_out, df, area = (
            x_in.to(self.device),
            x_out.to(self.device),
            df.to(self.device),
            area.to(self.device),
        )

        return x_in, x_out, df, area

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict)
        x_in = x_in[:: self.subsample_eval, ...]
        area = area[:: self.subsample_eval]

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            pred_chunks = []
            x_in_chunks = torch.split(x_in, r, dim=0)
            area_chunks = torch.split(area, r, dim=0)
            for j in range(len(x_in_chunks)):
                pred_chunks.append(
                    self(
                        x_in,
                        x_out,
                        df,
                        x_in_chunks[j],
                        area_in=area,
                        area_eval=area_chunks[j],
                    )
                )
            pred = torch.cat(tuple(pred_chunks), dim=0)
        else:
            pred = self(x_in, x_out, df, area=area)

        pred = pred.reshape(1, -1)

        if loss_fn is None:
            loss_fn = self.loss
        truth = (
            data_dict["pressure"][0]
            .to(self.device)[:: self.subsample_eval, ...]
            .reshape(1, -1)
        )
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict)
        x_in = x_in[:: self.subsample_train, ...]
        area = area[:: self.subsample_train]

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            indices = torch.randperm(x_in.shape[0])[:r]
            pred = self(
                x_in,
                x_out,
                df,
                x_in[indices, ...],
                area_in=area,
                area_eval=area[indices],
            )
        else:
            pred = self(x_in, x_out, df)

        if loss_fn is None:
            loss_fn = self.loss

        truth = data_dict["pressure"][0][:: self.subsample_train]
        if self.max_in_points is not None:
            truth = truth[indices].view(1, -1).to(self.device)
        else:
            truth = truth.view(1, -1).to(self.device)

        return {"loss": loss_fn(pred.view(1, -1), truth)}


class GNOFNOGNOTrackB(GNOFNOGNO):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=16,
        hidden_channels=(86, 86),
        in_channels=2,
        out_channels=1,
        fno_modes=(32, 32, 32),
        fno_hidden_channels=86,
        fno_out_channels=86,
        fno_domain_padding=0.125,
        fno_norm="ada_in",
        adain_embed_dim=64,
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
        max_in_points=None,
        subsample_train=1,
        subsample_eval=1,
    ):
        if fno_norm == "ada_in":
            init_norm = "group_norm"
        else:
            init_norm = fno_norm

        self.max_in_points = max_in_points
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval
        print("radius_in", radius_in, "radius_out", radius_out)
        super().__init__(
            radius_in=radius_in,
            radius_out=radius_out,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_out_channels=fno_out_channels,
            fno_domain_padding=fno_domain_padding,
            fno_norm=init_norm,
            fno_factorization=fno_factorization,
            fno_rank=fno_rank,
            linear_kernel=linear_kernel,
            weighted_kernel=weighted_kernel,
        )

        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = nn.ModuleList(
                AdaIN(adain_embed_dim, fno_hidden_channels)
                for _ in range(
                    self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers
                )
            )
            self.use_adain = True
        else:
            self.use_adain = False

    def data_dict_to_input(self, data_dict):
        x_in = data_dict["centroids"][0]  # (n_in, 3)
        x_out = (
            data_dict["sdf_query_points"].squeeze(0).permute(1, 2, 3, 0)
        )  # (n_x, n_y, n_z, 3)
        df = data_dict["sdf"]  # (1, n_x, n_y, n_z)
        area = data_dict["areas"][0]  # (n_in, 3)

        # info_fields = torch.cat([
        #    v*torch.ones_like(df) for _, v in data_dict['info'][0].items()
        # ], dim=0) # (8, n_x, n_y, n_z)

        # info_fields = data_dict["info"][0]["velocity"] * torch.ones_like(df)

        # df = torch.cat((df, info_fields), dim=0)  # (2, n_x, n_y, n_z)

        if self.use_adain:
            vel = (
                torch.tensor([data_dict["info"][0]["velocity"]])
                .view(
                    -1,
                )
                .to(self.device)
            )
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)

        x_in, x_out, df, area = (
            x_in.to(self.device),
            x_out.to(self.device),
            df.to(self.device),
            area.to(self.device),
        )

        return x_in, x_out, df, area

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict)
        x_in = x_in[:: self.subsample_eval, ...]
        area = area[:: self.subsample_eval]

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            pred_chunks = []
            x_in_chunks = torch.split(x_in, r, dim=0)
            area_chunks = torch.split(area, r, dim=0)
            for j in range(len(x_in_chunks)):
                pred_chunks.append(
                    self(
                        x_in,
                        x_out,
                        df,
                        x_in_chunks[j],
                        area_in=area,
                        area_eval=area_chunks[j],
                    )
                )
            pred = torch.cat(tuple(pred_chunks), dim=0)
        else:
            pred = self(x_in, x_out, df, area_in=area)

        pred = pred.reshape(1, -1)

        if loss_fn is None:
            loss_fn = self.loss
        truth = (
            data_dict["pressure"][0]
            .to(self.device)[:: self.subsample_eval, ...]
            .reshape(1, -1)
        )
        out_dict = {"l2": loss_fn(pred, truth)}

        if decode_fn is not None:
            pred = decode_fn(pred)
            truth = decode_fn(truth)
            out_dict["l2_decoded"] = loss_fn(pred, truth)
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict)
        x_in = x_in[:: self.subsample_train, ...]
        area = area[:: self.subsample_train]

        if self.max_in_points is not None:
            r = min(self.max_in_points, x_in.shape[0])
            indices = torch.randperm(x_in.shape[0])[:r]
            pred = self(
                x_in,
                x_out,
                df,
                x_in[indices, ...],
                area_in=area,
                area_eval=area[indices],
            )
        else:
            pred = self(
                x_in=x_in, x_out=x_out, df=df, x_eval=None, area_in=area, area_eval=None
            )

        if loss_fn is None:
            loss_fn = self.loss

        truth = data_dict["pressure"][0][:: self.subsample_train]
        if self.max_in_points is not None:
            truth = truth[indices].view(1, -1).to(self.device)
        else:
            truth = truth.view(1, -1).to(self.device)

        return {"loss": loss_fn(pred.view(1, -1), truth)}

    # def loss_dict(self, data_dict, loss_fn=None):
    #     x_in, x_out, df, area = self.data_dict_to_input(data_dict)
    #     sampled_x_in = [x_in[i::self.subsample_train] for i in range(self.subsample_train)]
    #     sampled_areas = [area[i::self.subsample_train] for i in range(self.subsample_train)]

    #     if self.max_in_points is not None:
    #         r = min(self.max_in_points, x_in.shape[0])
    #         indices = torch.randperm(x_in.shape[0])[:r]
    #         pred = self(
    #             x_in,
    #             x_out,
    #             df,
    #             x_in[indices, ...],
    #             area_in=area,
    #             area_eval=area[indices],
    #         )
    #     else:
    #         final_output = torch.zeros(area.shape[0])
    #         for idx in range(len(sampled_x_in)):
    #             pred_mini = self(x_in=sampled_x_in[idx], x_out=x_out, df=df, x_eval=None, area_in=sampled_areas[idx], area_eval=None).squeeze()
    #             final_output[idx::self.subsample_train] = pred_mini
    #         pred = final_output
    #     if loss_fn is None:
    #         loss_fn = self.loss

    #     truth = data_dict["pressure"][0][:: self.subsample_train]
    #     if self.max_in_points is not None:
    #         truth = truth[indices].view(1, -1).to(self.device)
    #     else:
    #         truth = truth.view(1, -1).to(self.device)

    #     return {"loss": loss_fn(pred.view(1, -1), truth)}

import argparse
import pickle as pk
import os
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from nltk.corpus import stopwords
import ipdb


class AutoEncoder(nn.Module):

    def __init__(self, input_dim1, input_dim2, hidden_dims, agg, sep_decode):
        super(AutoEncoder, self).__init__()

        self.agg = agg
        self.sep_decode = sep_decode

        print("hidden_dims:", hidden_dims)
        self.encoder_layers = []
        self.encoder2_layers = []
        dims = [[input_dim1, input_dim2]] + hidden_dims
        for i in range(len(dims) - 1):
            if i == 0:
                layer = nn.Sequential(nn.Linear(dims[i][0], dims[i+1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i][1], dims[i+1]), nn.ReLU())
            elif i != 0 and i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
                layer2 = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
            self.encoder2_layers.append(layer2)
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.encoder2 = nn.Sequential(*self.encoder2_layers)

        self.decoder_layers = []
        self.decoder2_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [[input_dim1, input_dim2]]
        if self.agg == "concat" and not self.sep_decode:
            dims[0] = 2 * dims[0]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1][0])
                layer2 = nn.Linear(dims[i], dims[i+1][1])
            self.decoder_layers.append(layer)
            self.decoder2_layers.append(layer2)
        self.decoder = nn.Sequential(*self.decoder_layers)
        self.decoder2 = nn.Sequential(*self.decoder2_layers)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder2(x2)

        if self.agg == "max":
            z = torch.max(z1, z2)
        elif self.agg == "multi":
            z = z1 * z2
        elif self.agg == "sum":
            z = z1 + z2
        elif self.agg == "concat":
            z = torch.cat([z1, z2], dim=1)

        if self.sep_decode:
            x_bar1 = self.decoder(z1)
            x_bar1 = F.normalize(x_bar1, dim=-1)
            x_bar2 = self.decoder2(z2)
            x_bar2 = F.normalize(x_bar2, dim=-1)
        else:
            x_bar1 = self.decoder(z)
            x_bar1 = F.normalize(x_bar1, dim=-1)
            x_bar2 = self.decoder2(z)
            x_bar2 = F.normalize(x_bar2, dim=-1)

        return x_bar1, x_bar2, z


class TopicCluster(nn.Module):

    def __init__(self, args):
        super(TopicCluster, self).__init__()
        self.alpha = 1.0
        self.dataset_path = args.dataset_path
        self.args = args
        self.device = args.device
        self.temperature = args.temperature
        self.distribution = args.distribution
        self.agg_method = args.agg_method
        self.sep_decode = (args.sep_decode == 1)

        input_dim1 = args.input_dim1
        input_dim2 = args.input_dim2
        hidden_dims = eval(args.hidden_dims)
        self.model = AutoEncoder(input_dim1, input_dim2, hidden_dims, self.agg_method, self.sep_decode)
        if self.agg_method == "concat":
            self.topic_emb = Parameter(torch.Tensor(args.n_clusters, 2*hidden_dims[-1]))
        else:
            self.topic_emb = Parameter(torch.Tensor(args.n_clusters, hidden_dims[-1]))
        torch.nn.init.xavier_normal_(self.topic_emb.data)

    def pretrain(self, input_data, pretrain_epoch=200):
        pretrained_path = os.path.join(self.dataset_path, f"pretrained_{args.suffix}.pt")
        if os.path.exists(pretrained_path) and self.args.load_pretrain:
            # load pretrain weights
            print(f"loading pretrained model from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))
        else:
            train_loader = DataLoader(input_data, batch_size=self.args.batch_size, shuffle=True)
            optimizer = Adam(self.model.parameters(), lr=self.args.lr)
            for epoch in range(pretrain_epoch):
                total_loss = 0
                for batch_idx, (x1, x2, _, weight) in enumerate(train_loader):
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    weight = weight.to(self.device)
                    optimizer.zero_grad()
                    x_bar1, x_bar2, z = self.model(x1, x2)
                    loss = cosine_dist(x_bar1, x1) + cosine_dist(x_bar2, x2) #, weight)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print(f"epoch {epoch}: loss = {total_loss / (batch_idx+1):.4f}")
            torch.save(self.model.state_dict(), pretrained_path)
            print(f"model saved to {pretrained_path}")

    def cluster_assign(self, z):
        if self.distribution == 'student':
            p = 1.0 / (1.0 + torch.sum(
                torch.pow(z.unsqueeze(1) - self.topic_emb, 2), 2) / self.alpha)
            p = p.pow((self.alpha + 1.0) / 2.0)
            p = (p.t() / torch.sum(p, 1)).t()
        else:
            self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
            z = F.normalize(z, dim=-1)
            sim = torch.matmul(z, self.topic_emb.t()) / self.temperature
            p = F.softmax(sim, dim=-1)
        return p
    
    def forward(self, x1, x2):
        x_bar1, x_bar2, z = self.model(x1, x2)
        p = self.cluster_assign(z)
        return x_bar1, x_bar2, z, p

    def target_distribution(self, x1, x2, freq, method='all', top_num=0):
        _, _, z = self.model(x1, x2)
        p = self.cluster_assign(z).detach()
        if method == 'all':
            q = p**2 / (p * freq.unsqueeze(-1)).sum(dim=0)
            q = (q.t() / q.sum(dim=1)).t()
        elif method == 'top':
            assert top_num > 0
            q = p.clone()
            sim = torch.matmul(self.topic_emb, z.t())
            _, selected_idx = sim.topk(k=top_num, dim=-1)
            for i, topic_idx in enumerate(selected_idx):
                q[topic_idx] = 0
                q[topic_idx, i] = 1
        return p, q


def cosine_dist(x_bar, x, weight=None):
    if weight is None:
        weight = torch.ones(x.size(0), device=x.device)
    cos_sim = (x_bar * x).sum(-1)
    cos_dist = 1 - cos_sim
    cos_dist = (cos_dist * weight).sum() / weight.sum()
    return cos_dist


def train(args, emb_dict):
    # ipdb.set_trace()
    inv_vocab = {k: " ".join(v) for k, v in emb_dict["inv_vocab"].items()}
    vocab = {" ".join(k):v for k, v in emb_dict["vocab"].items()}
    print(f"Vocab size: {len(vocab)}")
    embs = F.normalize(torch.tensor(emb_dict["vs_emb"]), dim=-1)
    embs2 = F.normalize(torch.tensor(emb_dict["oh_emb"]), dim=-1)
    freq = np.array(emb_dict["tuple_freq"])
    if not args.use_freq:
        freq = np.ones_like(freq)

    input_data = TensorDataset(embs, embs2, torch.arange(embs.size(0)), torch.tensor(freq))
    topic_cluster = TopicCluster(args).to(args.device)
    topic_cluster.pretrain(input_data, args.pretrain_epoch)
    train_loader = DataLoader(input_data, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(topic_cluster.parameters(), lr=args.lr)

    # topic embedding initialization
    embs = embs.to(args.device)
    embs2 = embs2.to(args.device)
    x_bar1, x_bar2, z = topic_cluster.model(embs, embs2)
    z = F.normalize(z, dim=-1)

    print(f"Running K-Means for initialization")
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=5)
    if args.use_freq:
        y_pred = kmeans.fit_predict(z.data.cpu().numpy(), sample_weight=freq)
    else:
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    print(f"Finish K-Means")

    freq = torch.tensor(freq).to(args.device)
    
    y_pred_last = y_pred
    topic_cluster.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    topic_cluster.train()
    i = 0
    for epoch in range(50):
        if epoch % 5 == 0:
            _, _, z, p = topic_cluster(embs, embs2)
            z = F.normalize(z, dim=-1)
            topic_cluster.topic_emb.data = F.normalize(topic_cluster.topic_emb.data, dim=-1)
            if not os.path.exists(os.path.join(args.dataset_path, f"clusters_{args.suffix}")):
                os.makedirs(os.path.join(args.dataset_path, f"clusters_{args.suffix}"))
            embed_save_path = os.path.join(args.dataset_path, f"clusters_{args.suffix}/embed_{epoch}.pt")
            torch.save({
                "inv_vocab": emb_dict['inv_vocab'],
                "embed": z.detach().cpu().numpy(),
                "topic_embed": topic_cluster.topic_emb.detach().cpu().numpy(),
            }, embed_save_path)
            f = open(os.path.join(args.dataset_path, f"clusters_{args.suffix}/{epoch}.txt"), 'w')
            pred_cluster = p.argmax(-1)
            result_strings = []
            for j in range(args.n_clusters):
                if args.sort_method == 'discriminative':
                    word_idx = torch.arange(embs.size(0))[pred_cluster == j]
                    sorted_idx = torch.argsort(p[pred_cluster == j][:, j], descending=True)
                    word_idx = word_idx[sorted_idx]
                else:
                    sim = torch.matmul(topic_cluster.topic_emb[j], z.t())
                    _, word_idx = sim.topk(k=30, dim=-1)
                word_cluster = []
                freq_sum = 0
                for idx in word_idx:
                    freq_sum += freq[idx].item()
                    if inv_vocab[idx.item()] not in word_cluster:
                        word_cluster.append(inv_vocab[idx.item()])
                        if len(word_cluster) >= 10:
                            break
                result_strings.append((freq_sum, f"Topic {j} ({freq_sum}): " + ', '.join(word_cluster)+'\n'))
            result_strings = sorted(result_strings, key=lambda x: x[0], reverse=True)
            for result_string in result_strings:
                f.write(result_string[1])
                 
        for x1, x2, idx, weight in train_loader:
            
            if i % args.update_interval == 0:
                p, q = topic_cluster.target_distribution(embs, embs2, freq.clone().fill_(1), method='all', top_num=epoch+1)

                y_pred = p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

                if i > 0 and delta_label < args.tol:
                    print(f'delta_label {delta_label:.4f} < tol ({args.tol})')
                    print('Reached tolerance threshold. Stopping training.')
                    return None

            i += 1
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            idx = idx.to(args.device)
            weight = weight.to(args.device)

            x_bar1, x_bar2, _, p = topic_cluster(x1, x2)
            reconstr_loss = cosine_dist(x_bar1, x1) + cosine_dist(x_bar2, x2) #, weight)
            kl_loss = F.kl_div(p.log(), q[idx], reduction='none').sum(-1)
            kl_loss = (kl_loss * weight).sum() / weight.sum()
            loss = args.gamma * kl_loss + reconstr_loss
            if i % args.update_interval == 0:
                print(f"KL loss: {kl_loss}; Reconstruction loss: {reconstr_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return None


def filter_vocab(emb_dict):
    stop_words = set(stopwords.words('english'))
    new_inv_vocab = [w for w, _ in emb_dict['vocab'].items() if w not in stop_words and not w.startswith('##')]
    new_vocab = {w:i for i, w in enumerate(new_inv_vocab)}
    new_avg_emb = emb_dict['avg_emb'][[emb_dict['vocab'][w] for w in new_inv_vocab]]
    return {"avg_emb": new_avg_emb, "vocab": new_vocab, "inv_vocab": new_inv_vocab}


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python3 latent_space_clustering.py --dataset_path ./pandemic --input_emb_name po_tuple_features_all_svos.pk
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--input_emb_name', type=str)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_clusters', default=30, type=int)
    parser.add_argument('--input_dim1', default=1000, type=int)
    parser.add_argument('--input_dim2', default=1000, type=int)
    parser.add_argument('--agg_method', default="multi", choices=["sum", "multi", "concat", "attend"], type=str)
    parser.add_argument('--sep_decode', default=0, choices=[0, 1], type=int)
    parser.add_argument('--pretrain_epoch', default=100, type=int)
    parser.add_argument('--load_pretrain', default=False, action='store_true')
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--sort_method', default='generative', choices=['generative', 'discriminative'])
    parser.add_argument('--distribution', default='softmax', choices=['softmax', 'student'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--use_freq', default=False, action='store_true')
    parser.add_argument('--hidden_dims', default='[1000, 2000, 1000, 100]', type=str)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--gamma', default=5, type=float, help='weight of clustering loss')
    parser.add_argument('--update_interval', default=100, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print(args)
    with open(os.path.join(args.dataset_path, args.input_emb_name), "rb") as fin:
        emb_dict = pk.load(fin)

    candidate_idx = train(args, emb_dict)
    print(candidate_idx)
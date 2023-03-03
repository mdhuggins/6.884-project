import torch
import pickle as pkl

def get_embeddings_file(model, g, filename):
    nids = torch.arange(0, g.num_nodes())
    g.edata['norm'] = torch.zeros(g.num_edges(), 1)
    with torch.no_grad():
        model.eval()
        embeddings = model.rgcn(g, nids)
    with open(filename, 'wb') as handle:
        pkl.dump(embeddings, handle)


def create_embeddings(filename, model_path, graph_path):
    model = torch.load(model_path)
    with open(graph_path, 'rb') as handle:
        g = pkl.load(handle)
    get_embeddings_file(model, g, filename)
    with open(filename, 'rb') as handle:
        embeddings = pkl.load(handle)
    return embeddings


def get_graph_embeddings(model,graph):
    nids = torch.arange(0, graph.num_nodes())
    graph.edata['norm'] = torch.zeros(graph.num_edges(), 1)
    embeddings = model.rgcn(graph, nids)
    return embeddings


if __name__ == '__main__':
    emb = get_graph_embeddings(n=580428)
    print(emb.shape)

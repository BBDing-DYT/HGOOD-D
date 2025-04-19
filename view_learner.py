import torch
from torch.nn import Sequential, Linear, ReLU


class ViewLearner(torch.nn.Module):
	def __init__(self, type_learner, encoder, mlp_edge_model_dim=64):
		super(ViewLearner, self).__init__()

		self.type_learner = type_learner
		self.encoder = encoder
		self.input_dim = self.encoder.out_node_dim

		self.mlp_edge_model = Sequential(
			Linear(self.input_dim * 2, mlp_edge_model_dim),
			ReLU(),
			Linear(mlp_edge_model_dim, 1)
		)
		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, x, edge_index):

		node_emb = self.encoder(x, edge_index)

		src, dst = edge_index[0], edge_index[1]
		emb_src = node_emb[src]
		emb_dst = node_emb[dst]

		if self.type_learner in ['mlp', 'gnn']:
			edge_emb = torch.cat([emb_src, emb_dst], 1)
			edge_logits = self.mlp_edge_model(edge_emb)
		elif self.type_learner == 'att':
			edge_logits = torch.sum(torch.mul(emb_src, emb_dst), dim=-1)

		return edge_logits
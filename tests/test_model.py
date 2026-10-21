import torch

from rtrec.model import TwoTower


def _model():
    torch.manual_seed(0)
    return TwoTower(n_users=20, n_items=30, n_genres=5, n_artists=8, emb_dim=8, hidden=16, out_dim=12)


def test_towers_output_unit_norm_vectors():
    m = _model()
    u = m.encode_user(torch.arange(4), torch.rand(4, 5))
    i = m.encode_item(torch.arange(4), torch.randint(0, 5, (4,)), torch.randint(0, 8, (4,)))
    assert u.shape == (4, 12) and i.shape == (4, 12)
    assert torch.allclose(u.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(i.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_in_batch_loss_decreases_on_repeated_batch():
    m = _model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    users = torch.arange(10)
    items = torch.arange(10)
    genre_profile = torch.rand(10, 5)
    genre = torch.randint(0, 5, (30,))
    artist = torch.randint(0, 8, (30,))
    losses = []
    for _ in range(30):
        loss = m.in_batch_softmax_loss(m.encode_user(users, genre_profile), m.encode_item(items, genre[items], artist[items]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

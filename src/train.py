


from data import PhonemeDataset, make_collate_fn
from torch.utils.data import DataLoader


dataset = PhonemeDataset(...)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=make_collate_fn(dataset.pad_id))

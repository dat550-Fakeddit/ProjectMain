import torch


# Prepare dataset before feeding to the model and training
def collate_X_Y(batch):
    X, Y = zip(*batch)
    X = torch.stack(X)
    Y = torch.stack(Y)
    # X = torch.stack([val[0] for val in batch])
    # Y = torch.stack([val[1] for val in batch])
    return X, Y.to_dense()

def collate_X_Y_Z(batch):
    X,Y,Z = zip(*batch)
    X = torch.stack(X)
    Y = torch.stack(Y) 
    Z = torch.stack(Z)

    # X = torch.stack([val[0] for val in batch])
    # Y = torch.stack([val[1] for val in batch])
    # Z = torch.stack([val[2] for val in batch])
    return X,Y,Z



def img_collate_fn(batch):
    # Extract feature vectors and labels from batch
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Determine the maximum length of feature vectors in the batch
    max_length = max(len(f) for f in features)
    
    # Pad feature vectors to the maximum length
    padded_features = [torch.nn.functional.pad(torch.tensor(f), (0, max_length - len(f))) for f in features]
    
    # Convert padded features to a tensor
    padded_features = torch.stack(padded_features)
    
    return padded_features, labels


if __name__ == '__main__':
    pass
import torch
from torch import optim

def iterate(dl, model, optimizer, update=False, training=False, scheduler=None):
    running_loss = 0
    cnt = 0
    acc_cnt = 0
    for x, y in dl:
        cnt += len(y)
        with torch.set_grad_enabled(training):
            out = model(x)
            y = model.make_y(y)
            loss = model.losses(out, y)
        running_loss += loss
        if update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return running_loss/cnt

def train(train, val, model, optimizer, scheduler=None, num_epochs=10, model_path='model/weights/model.pt'):
    best_loss = float('Inf')
    cnt = 0 # counter for how many epochs with out val improvement
    for i in range(num_epochs):
        if i == 2:
            print('unfreezing backbone')
            for param in model.backbone.parameters():
                param.requires_grad=True
            for param in model.upsample.parameters():
                param.requires_grad=True
            optimizer = optim.Adam([{'params': model.backbone.parameters(), 'lr':1e-3/9},
                                    {'params': model.upsample.parameters(), 'lr':1e-3/6},
                                    {'params': model.head.parameters(), 'lr':1e-3}])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5)
        model.train()
        train_loss = iterate(train, model, optimizer, update=True, training=True, scheduler=scheduler)
        #model.eval()
        val_loss = iterate(val, model, optimizer, update=False)
        if scheduler:
            scheduler.step(val_loss)
        if val_loss < best_loss:
            cnt = 0
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
        print(f'Epoch {i}: train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')
        cnt += 1
        if cnt == 10:
            print('early stopping')
            break
            
    model.load_state_dict(torch.load(model_path))
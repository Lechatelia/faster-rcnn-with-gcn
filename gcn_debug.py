import  torch
from model.faster_rcnn.gcn import gcn
def debug():
    data = torch.load('input.pt')['input']
    data = [ d.cuda() for d in  data]
    fasterRCNN = gcn(torch.load('voc_classes.pt')['classes'], pretrained=True, class_agnostic=False)
    fasterRCNN.create_architecture()
    fasterRCNN.cuda()
    fasterRCNN.train()
    fasterRCNN(*data)


if __name__ == "__main__":
    debug()

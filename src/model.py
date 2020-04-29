import torch
from collections import OrderedDict

import torch
import torch.nn as nn

def make_layers(block, no_relu_layers,prelu_layers = []):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                if layer_name not in prelu_layers:
                    layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
                else:
                    layers.append(('prelu'+layer_name[4:],nn.PReLU(v[1])))

    return nn.Sequential(OrderedDict(layers))

def make_layers_Mconv(block,no_relu_layers):
    modules = []
    for layer_name, v in block.items():
        layers = []
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('Mprelu'+layer_name[5:], nn.PReLU(v[1])))
        modules.append(nn.Sequential(OrderedDict(layers)))
    return nn.ModuleList(modules)

class bodypose_25_model(nn.Module):
    def __init__(self):
        super(bodypose_25_model,self).__init__()
        # these layers have no relu layer
        no_relu_layers = ['Mconv7_stage0_L1','Mconv7_stage0_L2',\
                          'Mconv7_stage1_L1', 'Mconv7_stage1_L2',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L2']
        prelu_layers = ['conv4_2','conv4_3_CPM','conv4_4_CPM']
        blocks = {}
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])
        self.model0 = make_layers(block0, no_relu_layers,prelu_layers)
        
        #L2
        #stage0
        blocks['Mconv1_stage0_L2'] = OrderedDict([
                    ('Mconv1_stage0_L2_0',[128,96,3,1,1]),
                    ('Mconv1_stage0_L2_1',[96,96,3,1,1]),
                    ('Mconv1_stage0_L2_2',[96,96,3,1,1])
                    ])
        for i in range(2,6):
            blocks['Mconv%d_stage0_L2' % i] = OrderedDict([
                    ('Mconv%d_stage0_L2_0' % i,[288,96,3,1,1]),
                    ('Mconv%d_stage0_L2_1' % i,[96,96,3,1,1]),
                    ('Mconv%d_stage0_L2_2' % i,[96,96,3,1,1])
              ])
        blocks['Mconv6_7_stage0_L2'] = OrderedDict([
                    ('Mconv6_stage0_L2',[288, 256, 1,1,0]),
                    ('Mconv7_stage0_L2',[256,52,1,1,0])
              ])
        #stage1~3
        for s in range(1,4):
            blocks['Mconv1_stage%d_L2' % s] = OrderedDict([
                    ('Mconv1_stage%d_L2_0' % s,[180,128,3,1,1]),
                    ('Mconv1_stage%d_L2_1' % s,[128,128,3,1,1]),
                    ('Mconv1_stage%d_L2_2' % s,[128,128,3,1,1])
                ])
            for i in range(2,6):
                blocks['Mconv%d_stage%d_L2' % (i,s)] = OrderedDict([
                        ('Mconv%d_stage%d_L2_0' % (i,s) ,[384,128,3,1,1]),
                        ('Mconv%d_stage%d_L2_1' % (i,s) ,[128,128,3,1,1]),
                        ('Mconv%d_stage%d_L2_2' % (i,s) ,[128,128,3,1,1])
                    ])
            blocks['Mconv6_7_stage%d_L2' % s] = OrderedDict([
                    ('Mconv6_stage%d_L2' % s,[384,512,1,1,0]),
                    ('Mconv7_stage%d_L2' % s,[512,52,1,1,0])
                ])
        
        #L1
        #stage0
        blocks['Mconv1_stage0_L1'] = OrderedDict([
                ('Mconv1_stage0_L1_0',[180,96,3,1,1]),
                ('Mconv1_stage0_L1_1',[96,96,3,1,1]),
                ('Mconv1_stage0_L1_2',[96,96,3,1,1])
            ])
        for i in range(2,6):
            blocks['Mconv%d_stage0_L1' % i] = OrderedDict([
                    ('Mconv%d_stage0_L1_0' % i,[288,96,3,1,1]),
                    ('Mconv%d_stage0_L1_1' % i,[96,96,3,1,1]),
                    ('Mconv%d_stage0_L1_2' % i,[96,96,3,1,1])
              ])
        blocks['Mconv6_7_stage0_L1'] = OrderedDict([
                    ('Mconv6_stage0_L1',[288, 256, 1,1,0]),
                    ('Mconv7_stage0_L1',[256,26,1,1,0])
              ])
        #stage1
        blocks['Mconv1_stage1_L1'] = OrderedDict([
                ('Mconv1_stage1_L1_0',[206,128,3,1,1]),
                ('Mconv1_stage1_L1_1',[128,128,3,1,1]),
                ('Mconv1_stage1_L1_2',[128,128,3,1,1])
            ])
        for i in range(2,6):
            blocks['Mconv%d_stage1_L1' % i] = OrderedDict([
                    ('Mconv%d_stage1_L1_0' % i,[384,128,3,1,1]),
                    ('Mconv%d_stage1_L1_1' % i,[128,128,3,1,1]),
                    ('Mconv%d_stage1_L1_2' % i,[128,128,3,1,1])
                ])
        blocks['Mconv6_7_stage1_L1'] = OrderedDict([
                ('Mconv6_stage1_L1',[384,512,1,1,0]),
                ('Mconv7_stage1_L1',[512,26,1,1,0])
            ])
        
        for k in blocks.keys():
            blocks[k] = make_layers_Mconv(blocks[k], no_relu_layers)
        self.models = nn.ModuleDict(blocks)
        #self.model_L2_S0_mconv1 = blocks['Mconv1_stage0_L2']
            
        
    def _Mconv_forward(self,x,models):
        outs = []
        out = x
        for m in models:
            out = m(out)
            outs.append(out)
        return torch.cat(outs,1)
        
    def forward(self,x):
        out0 = self.model0(x)
        #L2
        tout = out0
        for s in range(4):
            tout = self._Mconv_forward(tout,self.models['Mconv1_stage%d_L2' % s])
            for v in range(2,6):
                tout = self._Mconv_forward(tout,self.models['Mconv%d_stage%d_L2' % (v,s)])
            tout = self.models['Mconv6_7_stage%d_L2' % s][0](tout)
            tout = self.models['Mconv6_7_stage%d_L2' % s][1](tout)
            outL2 = tout
            tout = torch.cat([out0,tout],1)
        #L1 stage0
        #tout = torch.cat([out0,outL2],1)
        tout = self._Mconv_forward(tout, self.models['Mconv1_stage0_L1'])
        for v in range(2,6):
            tout = self._Mconv_forward(tout, self.models['Mconv%d_stage0_L1' % v])
        tout = self.models['Mconv6_7_stage0_L1'][0](tout)
        tout = self.models['Mconv6_7_stage0_L1'][1](tout)
        outS0L1 = tout
        tout = torch.cat([out0,outS0L1,outL2],1)
        #L1 stage1
        tout = self._Mconv_forward(tout, self.models['Mconv1_stage1_L1'])
        for v in range(2,6):
            tout = self._Mconv_forward(tout, self.models['Mconv%d_stage1_L1' % v])
        tout = self.models['Mconv6_7_stage1_L1'][0](tout)
        outS1L1 = self.models['Mconv6_7_stage1_L1'][1](tout)
        
        return outS1L1,outL2
        

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])


        # Stage 1
        block1_1 = OrderedDict([
                        ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
                    ])

        block1_2 = OrderedDict([
                        ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
                    ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
                ])

            blocks['block%d_2' % i] = OrderedDict([
                    ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']


    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return  out6_2,out6_1

class handpose_model(nn.Module):
    def __init__(self):
        super(handpose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict([
                ('conv1_1', [3, 64, 3, 1, 1]),
                ('conv1_2', [64, 64, 3, 1, 1]),
                ('pool1_stage1', [2, 2, 0]),
                ('conv2_1', [64, 128, 3, 1, 1]),
                ('conv2_2', [128, 128, 3, 1, 1]),
                ('pool2_stage1', [2, 2, 0]),
                ('conv3_1', [128, 256, 3, 1, 1]),
                ('conv3_2', [256, 256, 3, 1, 1]),
                ('conv3_3', [256, 256, 3, 1, 1]),
                ('conv3_4', [256, 256, 3, 1, 1]),
                ('pool3_stage1', [2, 2, 0]),
                ('conv4_1', [256, 512, 3, 1, 1]),
                ('conv4_2', [512, 512, 3, 1, 1]),
                ('conv4_3', [512, 512, 3, 1, 1]),
                ('conv4_4', [512, 512, 3, 1, 1]),
                ('conv5_1', [512, 512, 3, 1, 1]),
                ('conv5_2', [512, 512, 3, 1, 1]),
                ('conv5_3_CPM', [512, 128, 3, 1, 1])
            ])

        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0])
        ])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                    ('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                    ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6



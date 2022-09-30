import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return ``input``, ``backward`` will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -0.1*coeff * grad_outputs


class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step.item())
        if self.training:
            self.global_step += 1.0
        return self.grl(self.coeff, x)

class Discriminator_hidden(nn.Module):
    def __init__(self, fingerprint_dim, max_iter=10000):
        super(Discriminator_hidden, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(fingerprint_dim, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100,100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=max_iter))
    def forward(self, mol_feature, reverse=False):
        if reverse:
            mol_feature = self.grl(mol_feature)
        output = self.main(mol_feature)
        return output
    
class Classifier(nn.Module):
    def __init__(self, fingerprint_dim, classes=2):
        super(Classifier, self).__init__()
        self.output = nn.Linear(fingerprint_dim, classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, mol_feature, reverse=False):
        output = self.output(mol_feature)
        output = self.softmax(output)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, fingerprint_dim, max_iter=10000):
        super(Discriminator, self).__init__()
        self.output = nn.Linear(fingerprint_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=1, max_iter=max_iter))
    def forward(self, mol_feature, reverse=False):
        if reverse:
            mol_feature = self.grl(mol_feature)
        output = self.output(mol_feature)
        output = self.sigmoid(output)
        return output

def normalize_weight(x, cut=0, expend=False, numpy=False):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    if expend:
        mean_val = torch.mean(x)
        x = x / mean_val
        x = torch.where(x >= cut, x, torch.zeros_like(x))
    if numpy:
        x = variable_to_numpy(x)
    return x.detach()


class AFSE(nn.Module):
    def __init__(self, fingerprint_dim, output_units_num, p_dropout):
        super(AFSE, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        self.metric = nn.Linear(2*fingerprint_dim, fingerprint_dim)
        self.output = nn.Linear(fingerprint_dim, output_units_num)
        self.fingerprint_dim = fingerprint_dim
#         self.BN = nn.BatchNorm1d(fingerprint_dim)
        
    def forward(self,feature=None, d=0, output_lr=False, sigmoid=False):
#         feature = self.BN(feature)
        unit_vector = torch.ones(self.fingerprint_dim).cuda()
        zero_vector = torch.zeros(self.fingerprint_dim).cuda()
        aW1 = self.output(self.dropout(self.metric(self.dropout(torch.cat((unit_vector, zero_vector),-1)))))
        aW2 = self.output(self.dropout(self.metric(self.dropout(torch.cat((zero_vector, unit_vector),-1)))))
        beta_AFSE = torch.norm(24*aW2**2/(aW1+aW2))**2
        beta_MSE = torch.norm(2*(aW1**2+aW2**2))**2
        conv_lr = 1/(beta_AFSE+beta_MSE+1e-9)
        mol_prediction = self.output(self.dropout(self.metric(self.dropout(torch.cat((feature, feature+d),-1)))))
        if sigmoid:
            mol_prediction = torch.sigmoid(mol_prediction)
        if output_lr:
            return mol_prediction, conv_lr
        return mol_prediction
    
    
# class AFSE(nn.Module):
#     def __init__(self, fingerprint_dim, output_units_num, p_dropout):
#         super(AFSE, self).__init__()
#         self.dropout = nn.Dropout(p=p_dropout)
#         self.metric = nn.Linear(2*fingerprint_dim, fingerprint_dim)
#         self.hidden = nn.Linear(fingerprint_dim, int(fingerprint_dim/1))
#         self.output = nn.Linear(int(fingerprint_dim/1), output_units_num)
#         self.fingerprint_dim = fingerprint_dim
# #         self.BN = nn.BatchNorm1d(fingerprint_dim)
        
#     def forward(self,feature=None, d=0, output_lr=False, sigmoid=False):
# #         feature = self.BN(feature)
#         unit_vector = torch.ones(self.fingerprint_dim).cuda()
#         zero_vector = torch.zeros(self.fingerprint_dim).cuda()
#         aW1 = self.output(self.dropout(self.hidden(self.dropout(self.metric(self.dropout(torch.cat((unit_vector, zero_vector),-1)))))))
#         aW2 = self.output(self.dropout(self.hidden(self.dropout(self.metric(self.dropout(torch.cat((zero_vector, unit_vector),-1)))))))
#         beta_AFSE = torch.norm(24*aW2**2/(aW1+aW2))**2
#         beta_MSE = torch.norm(2*(aW1**2+aW2**2))**2
#         conv_lr = 1/(beta_AFSE+beta_MSE+1e-9)
#         mol_prediction = self.output(self.dropout(self.hidden(self.dropout(self.metric(self.dropout(torch.cat((feature, feature+d),-1)))))))
#         if sigmoid:
#             mol_prediction = torch.sigmoid(mol_prediction)
#         if output_lr:
#             return mol_prediction, conv_lr
#         return mol_prediction

    
# class AFSE(nn.Module):
#     def __init__(self, fingerprint_dim, output_units_num, p_dropout):
#         super(AFSE, self).__init__()
#         self.dropout = nn.Dropout(p=p_dropout)
# #         self.attend = nn.ModuleList([nn.Linear(2*fingerprint_dim, 1) for r in range(output_units_num)])
#         self.metric = nn.ModuleList([nn.Linear(2*fingerprint_dim, fingerprint_dim) for r in range(output_units_num)])
#         self.output = nn.ModuleList([nn.Linear(fingerprint_dim, 1) for r in range(output_units_num)])
#         self.fingerprint_dim = fingerprint_dim
#         self.output_units_num = output_units_num
# #         self.BN = nn.BatchNorm1d(fingerprint_dim)
        
#     def forward(self,feature=None, d=0, output_lr=False, sigmoid=False):
# #         feature = self.BN(feature)
#         conv_lr_list = []
#         mol_prediction = torch.tensor([]).cuda()
#         unit_vector = torch.ones(self.fingerprint_dim).cuda()
#         zero_vector = torch.zeros(self.fingerprint_dim).cuda()
#         for i in range(self.output_units_num):
#             aW1 = self.output[i](self.dropout(self.metric[i](self.dropout(torch.cat((unit_vector, zero_vector),-1)))))
#             aW2 = self.output[i](self.dropout(self.metric[i](self.dropout(torch.cat((zero_vector, unit_vector),-1)))))
#             beta_AFSE = torch.norm(24*aW2**2/(aW1+aW2))**2
#             beta_MSE = torch.norm(2*(aW1**2+aW2**2))**2
#             conv_lr = 1/(beta_AFSE+beta_MSE+1e-9)
#             conv_lr_list.append(conv_lr.detach())
#             task_mol_prediction = self.output[i](self.dropout(self.metric[i](self.dropout(torch.cat((feature, feature+d),-1))))) #self.attend[i](torch.cat((feature, feature+d),-1))*
#             mol_prediction = torch.cat([mol_prediction, task_mol_prediction], -1)
        
#         if sigmoid:
#             mol_prediction = torch.sigmoid(mol_prediction)
#         if output_lr:
#             return mol_prediction, conv_lr_list
#         return mol_prediction
   

    
class Fingerprint_MutiTask(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout, task_num):
        super(Fingerprint_MutiTask, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for i in range(task_num)])
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        self.radius = radius
        self.T = T
        self.task_num = task_num


    def forward(self,atom_list=None,bond_list=None,atom_degree_list=None,bond_degree_list=None,atom_mask=None,output_activated_features=False):
        
        atom_mask = atom_mask.unsqueeze(2)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
#             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
        attention_weight = attention_weight * attend_mask
#         print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
#             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
#             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        #do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))
    #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
            attention_weight = attention_weight * attend_mask
#             print(attention_weight)
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
    #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
    #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
#             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            
            # do nonlinearity
            activated_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        
        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)           
        
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        activated_features_mol_list = []
        for i in range(self.task_num):
            for t in range(self.T):
                mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
                mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
                mol_align_score = F.leaky_relu(self.mol_align[i](mol_align))
                mol_align_score = mol_align_score + mol_softmax_mask
                mol_attention_weight = F.softmax(mol_align_score,-2)
                mol_attention_weight = mol_attention_weight * atom_mask
    #             print(mol_attention_weight.shape,mol_attention_weight)
                activated_features_transform = self.mol_attend(self.dropout(activated_features))
    #             aggregate embeddings of atoms in a molecule
                mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
    #             print(mol_context.shape,mol_context)
                mol_context = F.elu(mol_context)
                mol_feature = self.mol_GRUCell(mol_context, mol_feature)
                # print('mol_feature:',mol_feature.shape,mol_feature)

                # do nonlinearity
                activated_features_mol = F.relu(mol_feature)           
            activated_features_mol_list.append(activated_features_mol)

        if output_activated_features:
            return activated_features, activated_features_mol_list
        return activated_features_mol_list
    
    
class Fingerprint(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout):
        super(Fingerprint, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim,1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        self.radius = radius
        self.T = T


    def forward(self,atom_list=None,bond_list=None,atom_degree_list=None,bond_degree_list=None,atom_mask=None,output_activated_features=False):
        
        atom_mask = atom_mask.unsqueeze(2)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
#             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
        attention_weight = attention_weight * attend_mask
#         print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
#             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
#             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        #do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))
    #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
            attention_weight = attention_weight * attend_mask
#             print(attention_weight)
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
    #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
    #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
#             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            
            # do nonlinearity
            activated_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        
        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)           
        
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        for t in range(self.T):
            
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_mask
#             print(mol_attention_weight.shape,mol_attention_weight)
            activated_features_transform = self.mol_attend(self.dropout(activated_features))
#             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
#             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            # print('mol_feature:',mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)           
        
        if output_activated_features:
            return activated_features, activated_features_mol
        return activated_features_mol

    
class GRN(nn.Module):

    def __init__(self, radius, T, output_atom_dim, output_bond_dim, fingerprint_dim, p_dropout):
        super(GRN, self).__init__()
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(fingerprint_dim, output_atom_dim)
        self.bond_fc = nn.Linear(2*fingerprint_dim, output_bond_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,fingerprint_dim) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim,fingerprint_dim)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.radius = radius
        self.T = T


    def forward(self,atom_list=None,bond_list=None,atom_degree_list=None,bond_degree_list=None,atom_mask=None, mol_feature=None, activated_features=None):
        fingerprint_dim = mol_feature.shape[-1]
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
#         print(atom_neighbor.shape,bond_neighbor.shape)
        batch_size, mol_length, max_neighbor_num, num_atom_feat = atom_neighbor.shape
        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)
        
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        mol_length = activated_features.shape[-2]
        mol_feature_expand = mol_feature.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
        mapping_weight = torch.mul(mol_feature_expand, activated_features).sum(-1)
        softmax = nn.Softmax(dim=-1)
        mapping_weight_normalized = softmax(mapping_weight)
        mapping_weight_expend = mapping_weight_normalized.unsqueeze(-1).expand(batch_size, mol_length, fingerprint_dim)
        git_feature = mapping_weight_expend * mol_feature_expand + activated_features

        for t in range(self.T):           
            mol_context = torch.cat([mol_feature_expand, git_feature], dim=-1)
            mol_context = F.elu(self.mol_align(self.dropout(mol_context)))
#             print(mol_context.shape,giT_features.shape)
            git_feature = self.mol_GRUCell(mol_context.view(batch_size*mol_length,fingerprint_dim), git_feature.view(batch_size*mol_length,fingerprint_dim)).view(batch_size,mol_length,fingerprint_dim)
            # print('mol_feature:',mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_atom_features = F.relu(git_feature)  
        
        for d in range(self.radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_atom_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_atom_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d](self.dropout(feature_align)))
    #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
#             print(attention_weight)
            attention_weight = attention_weight * attend_mask
#             print(attention_weight)
            neighbor_feature_transform = self.attend[d](self.dropout(neighbor_feature))
    #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
    #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = activated_atom_features.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
            
            # do nonlinearity
            activated_atom_features = F.relu(atom_feature)

        atom_list = self.atom_fc(activated_atom_features)
        atom_neighbor = [activated_atom_features[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        atom_features_expend = activated_atom_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        
        bond_feature = torch.cat([atom_features_expend, atom_neighbor],dim=-1)
        bond_list = self.bond_fc(bond_feature)
        
        # activate for one hot type of initial feature
        atom_one_hot_mask = -9e8*torch.ones_like(atom_list, requires_grad=False)
        atom_one_hot_mask[:,:,:16] = 0
        a1 = softmax(atom_list+atom_one_hot_mask)
        mask = torch.zeros_like(atom_list, requires_grad=False)
        mask[:,:,:16] = 1
        a1 = a1*mask
        atom_one_hot_mask = -9e8*torch.ones_like(atom_list, requires_grad=False)
        atom_one_hot_mask[:,:,16:22] = 0
        a2 = softmax(atom_list+atom_one_hot_mask)
        mask = torch.zeros_like(atom_list, requires_grad=False)
        mask[:,:,16:22] = 1
        a2 = a2*mask
        atom_one_hot_mask = -9e8*torch.ones_like(atom_list, requires_grad=False)
        atom_one_hot_mask[:,:,24:30] = 0
        a3 = softmax(atom_list+atom_one_hot_mask)
        mask = torch.zeros_like(atom_list, requires_grad=False)
        mask[:,:,24:30] = 1
        a3 = a3*mask
        atom_one_hot_mask = -9e8*torch.ones_like(atom_list, requires_grad=False)
        atom_one_hot_mask[:,:,31:36] = 0
        a4 = softmax(atom_list+atom_one_hot_mask)
        mask = torch.zeros_like(atom_list, requires_grad=False)
        mask[:,:,31:36] = 1
        a4 = a4*mask
        bond_one_hot_mask = -9e8*torch.ones_like(bond_list, requires_grad=False)
        bond_one_hot_mask[:,:,:,:4] = 0
        b1 = softmax(bond_list+bond_one_hot_mask)
        mask = torch.zeros_like(bond_list, requires_grad=False)
        mask[:,:,:,:4] = 1
        b1 = b1*mask
        bond_one_hot_mask = -9e8*torch.ones_like(bond_list, requires_grad=False)
        bond_one_hot_mask[:,:,:,6:10] = 0
        b2 = softmax(bond_list+bond_one_hot_mask)
        mask = torch.zeros_like(bond_list, requires_grad=False)
        mask[:,:,:,6:10] = 1
        b2 = b2*mask
        # activate for interger type of initial feature
        atom_interger_mask = torch.zeros_like(atom_list, requires_grad=False)
        atom_interger_mask[:,:,24] = 1
        a5 = F.relu(atom_list*atom_interger_mask)*atom_interger_mask
        # activate for binary type of initial feature
        atom_binary_mask = torch.zeros_like(atom_list, requires_grad=False)
        atom_binary_mask[:,:,30] = 1
        a6 = torch.sigmoid(atom_list*atom_binary_mask)*atom_binary_mask
        atom_binary_mask = torch.zeros_like(atom_list, requires_grad=False)
        atom_binary_mask[:,:,36] = 1
        a7 = torch.sigmoid(atom_list*atom_binary_mask)*atom_binary_mask
        atom_binary_mask = torch.zeros_like(atom_list, requires_grad=False)
        atom_binary_mask[:,:,37] = 1
        a8 = torch.sigmoid(atom_list*atom_binary_mask)*atom_binary_mask
        atom_binary_mask = torch.zeros_like(atom_list, requires_grad=False)
        atom_binary_mask[:,:,38] = 1
        a9 = torch.sigmoid(atom_list*atom_binary_mask)*atom_binary_mask
        bond_binary_mask = torch.zeros_like(bond_list, requires_grad=False)
        bond_binary_mask[:,:,:,4] = 1
        b3 = torch.sigmoid(bond_list*bond_binary_mask)*bond_binary_mask
        bond_binary_mask = torch.zeros_like(bond_list, requires_grad=False)
        bond_binary_mask[:,:,:,5] = 1
        b4 = torch.sigmoid(bond_list*bond_binary_mask)*bond_binary_mask
        
        activate_atom_list = a1+a2+a3+a4+a5+a6+a7+a8+a9
        activate_bond_list = b1+b2+b3+b4
        return activate_atom_list, activate_bond_list
    
    
class Fingerprint_viz(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout):
        super(Fingerprint_viz, self).__init__()
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim,1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        
        self.dropout = nn.Dropout(p=p_dropout)
        self.metric = nn.Linear(2*fingerprint_dim, fingerprint_dim)
        self.output = nn.Linear(fingerprint_dim, output_units_num)
        self.shared = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.shared_output = nn.Linear(2*fingerprint_dim, output_units_num)

        self.radius = radius
        self.T = T


    def forward(self,atom_list=None,bond_list=None,atom_degree_list=None,bond_degree_list=None,atom_mask=None,output_feature=False,feature_only=False,feature1=None,feature2=None,d=0, viz=False,output_activated_features=False):
        if feature_only:
            mol_prediction = self.output(self.dropout(self.metric(self.dropout(torch.cat((feature1,feature2),-1)))))
            return mol_prediction
        atom_mask = atom_mask.unsqueeze(2)
        batch_size,mol_length,num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))

        atom_feature_viz = []
        atom_feature_viz.append(self.atom_fc(atom_list))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        #then catenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_attention = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        
        align_score = self.dropout(F.leaky_relu(self.align[0](feature_attention)))
#             print(align_score)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
#             print(align_score)
        attention_weight = attention_weight * attend_mask
#         print(align_score)
        atom_attention_weight_viz = []
        atom_attention_weight_viz.append(attention_weight)

        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
#             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
#             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
#         atom_feature_viz.append(atom_feature)

        #do nonlinearity
        activated_features = F.relu(atom_feature)
        atom_feature_viz.append(activated_features)
        
        for d in range(self.radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_attention = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = self.dropout(F.leaky_relu(self.align[d+1](feature_attention)))
    #             print(align_score)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)
#             print(align_score)
            attention_weight = attention_weight * attend_mask
            atom_attention_weight_viz.append(attention_weight)
    
#             print(align_score)
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
    #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
    #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
#             atom_feature_viz.append(atom_feature)
            
            #do nonlinearity
            activated_features = F.relu(atom_feature)
            atom_feature_viz.append(activated_features)

        
        # when the descriptor value are unbounded, like partial charge or LogP
        mol_feature_unbounded_viz = []
        mol_feature_unbounded_viz.append(torch.sum(atom_feature * atom_mask, dim=-2)) 
        
        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        #do nonlinearity
        activated_features_mol = F.relu(mol_feature)
        
        # when the descriptor value has lower or upper bounds
        mol_feature_viz = []
        mol_feature_viz.append(mol_feature) 
        
        mol_attention_weight_viz = []
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        
        for t in range(self.T):
            
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = self.dropout(F.leaky_relu(self.mol_align(mol_align)))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_mask
#             print(mol_attention_weight.shape,mol_attention_weight)
            mol_attention_weight_viz.append(mol_attention_weight)

            activated_features_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
#             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
#             print(mol_feature.shape,mol_feature)

            mol_feature_unbounded_viz.append(mol_feature)
            #do nonlinearity
            activated_features_mol = F.relu(mol_feature)           
            mol_feature_viz.append(activated_features_mol)
            
        mol_prediction = self.output(self.dropout(self.metric(self.dropout(torch.cat((mol_feature,mol_feature+d),-1)))))
        if viz:
            return atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, mol_attention_weight_viz, mol_prediction
        if output_feature:
            return mol_feature, mol_prediction
        if output_activated_features:
            return activated_features, mol_feature, mol_prediction
        return mol_prediction
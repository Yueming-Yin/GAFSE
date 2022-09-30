# GAFSE
Reproducing of the paper entitled "An open unified deep graph learning framework for discovering drug leads" (Nature Machine Intelligence, Submitted, IF: 25.898)

- All rights reserved by Yueming Yin, Email: 1018010514@njupt.edu.cn (or yinym96@qq.com).

# Environment
- Python 3.6.13
- Pytorch 1.10.2 (Up to your CUDA version)
- RDKit 2020.09.1.0
- Jupyter
- Tensorboard
- Tensorboardx
- Seaborn
- Matplotlib
- Scikit-learn

# Reproduction on GAFSE-HS (Hit screening)
In your jupyter notebook, re-run the corresponding notebook of "GAFSE-HS/GAFSE-HS/1_GAFSE_{Task ID}.ipynb" to reproduce the training process of GAFSE-HS. For example:
```
./GAFSE-HS/GAFSE-HS/1_GAFSE_1.ipynb
```
To reproduce the training process of GAFSE-MO on these HS tasks, please re-run the corresponding notebook of "GAFSE-HS/GAFSE-MO/1_GAFSE_{Task ID}.ipynb". For example:
```
./GAFSE-HS/GAFSE-MO/3_GAFSE_1.ipynb
```

# Reproduction on GAFSE-MP (Molecular property prediction)
In your jupyter notebook, re-run the corresponding notebook of "GAFSE-MP/3C_GAFSE_Multi_Tasks_{Small, Medium, Big, Large}.ipynb" to reproduce the training process of GAFSE-MP. For example:
```
./GAFSE-MP/3C_GAFSE_Multi_Tasks_Small.ipynb
```
To readout the GAFSE-MP performance over all tasks, please re-run the notebook of "GAFSE-MP/3C_Performance_Readout.ipynb":
```
./GAFSE-MP/3C_Performance_Readout.ipynb
```

# Reproduction on GAFSE-MO (Molecule optimization)
## Reproduce GAFSE-MO on Molecular Activity
In your jupyter notebook, re-run the corresponding notebook of "GAFSE-MO/G_AFSE_{Task name}_1/G_AFSE_{Task name}_1.ipynb" to reproduce the training process of GAFSE-MO on molecular activity. For example:
```
./GAFSE-MO/G_AFSE_IC50_O43614_1/G_AFSE_IC50_O43614_1.ipynb
```
To test trained models, please re-run the corresponding notebook of "GAFSE-MO/G_AFSE_{Task name}_1/G_AFSE_{Task name}_1-Test.ipynb". For example:
```
./GAFSE-MO/G_AFSE_IC50_O43614_1/G_AFSE_IC50_O43614_1-Test.ipynb
```

## Reproduce GAFSE-MO on Molecular Property
In your jupyter notebook, re-run the corresponding notebook of "GAFSE-MO/G_ADMET_M,T_C_{Task name}/G_ADMET_M,T_C_{Task name}.ipynb" to reproduce the training process of GAFSE-MO on molecular property. For example:
```
./GAFSE-MO/G_ADMET_M_C_CYP1A2_inhibitor/G_ADMET_M_C_CYP1A2_inhibitor.ipynb
```
To test trained models, please re-run the corresponding notebook of "GAFSE-MO/G_ADMET_M,T_C_{Task name}/G_ADMET_M,T_C_{Task name}-Test.ipynb". For example:
```
./GAFSE-MO/G_ADMET_M_C_CYP1A2_inhibitor/G_ADMET_M_C_CYP1A2_inhibitor-Test.ipynb
```

## Reproduce GAFSE-MO on COVID-19
In your jupyter notebook, re-run the notebook of "GAFSE-MO/AID_1706/G_AFSE_AID_1706.ipynb" to reproduce the training process of GAFSE-MO on the COVID-19-related database:
```
./GAFSE-MO/AID_1706/G_AFSE_AID_1706.ipynb
```
For a quick view of the results, please check "GAFSE-MO/AID_1706/AID_1706_datatable_copy_generated_molecules.csv"

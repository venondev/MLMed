import os
import nibabel as nib

datapath = "/media/lm/Samsung_T5/Uni/Medml/training/test/test_out_rescaled"
datapath_final = "/media/lm/Samsung_T5/Uni/Medml/training/test/final_out"

if not os.path.exists(datapath_final):
    os.makedirs(datapath_final)

files = os.listdir(datapath)
print(files)

for idx, file in enumerate(files):
    print(f"{idx + 1}/{len(files)} : {file}")

    sum_ = nib.load(datapath + "/" + file)

    pred_data = sum_.get_fdata()

    print(pred_data.max())

    result = pred_data >= 1.5

    print(result.sum())

    nib.save(nib.Nifti1Image(result.astype(int), sum_.affine, sum_.header), datapath_final + "/" + file + "_pred_final.nii.gz")




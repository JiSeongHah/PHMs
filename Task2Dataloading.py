from scipy import io

# fle = io.loadmat('/home/a286winteriscoming/Downloads/MATLAB_Code (anomaly detection)/saved_data/pca.mat')
fle = io.loadmat('/home/a286winteriscoming/Downloads/MATLAB_Code (anomaly detection)/input_data/outerrace_fault.mat')
# fle = io.loadmat('/home/a286winteriscoming/Downloads/MATLAB_Code (anomaly detection)/saved_data/Featureset_Outer.mat')


print(fle['outer'].shape)



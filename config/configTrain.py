"""User Model Global Config"""
model_name = 'df_z32_c1_dit_n11_mario_km_tanh_ldm'
action_space = 7
model_path = "./model.pth"
device = 'cuda:0'

"""Local Web"""
file_path= '../eval_data/0-frameArray.txt'
data_type='java'
SEQ_LEN = 100


"""Train Config"""
img_size = 128
img_channel = 3

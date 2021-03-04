MODEL_CONFIGS = {
    "PROJ_ROOT": '/Users/orekinana/Desktop/traffic_speed_estimation_and_anomaly_detection',
    # "PROJ_ROOT": '/data/yuanxun/traffic',
    "JN2": #
    {
        'node_num': 608,
        're_hidden': [608,500,100,50,10],
        'dis_hidden': [608,500,100,50,10],
        'vi_feature': 10,
        'kernel_size': 12,
        'alphi1': 1,
        'alphi2': 1,
        'alphi3': 0.0001,
    },

    "JNv": #
    {
        'node_num': 608,
        're_hidden': [608,500,50,5],
        'dis_hidden': [608,500,50,5],
        'vi_feature': 5,
        'kernel_size': 12,
        'alphi1': 1,
        'alphi2': 1,
        'alphi3': 0.0002,
    },

}
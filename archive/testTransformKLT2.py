# raw data from full_seq_1 frame 2235
# Raw Roam Info:
# Timestamp: 1547121346397879
# EST Pose: [1049.133,-304.933,49.911°]
# GT Deltas: [0.438,0.096,5.212°]
# EST Deltas: [1.315,0.000,0.858°]

# As you can see the est deltas is much too low
# :(

import numpy as np
from getTransformKLT import *
from genFakeData import *

srcCoord = np.array(
    [
        [832.0, 1186.0],
        [946.0, 952.0],
        [887.0, 1008.0],
        [820.0, 1033.0],
        [1177.0, 1358.0],
        [554.0, 1061.0],
        [796.0, 1149.0],
        [1111.0, 1019.0],
        [711.0, 1195.0],
        [1318.0, 941.0],
        [979.0, 948.0],
        [1178.0, 906.0],
        [1139.0, 930.0],
        [1259.0, 793.0],
        [869.0, 1170.0],
        [1158.0, 820.0],
        [1184.0, 1090.0],
        [1162.0, 996.0],
        [1195.0, 870.0],
        [1076.0, 674.0],
        [991.0, 1106.0],
        [882.0, 1127.0],
        [760.0, 1041.0],
        [616.0, 1047.0],
        [1500.0, 941.0],
        [1117.0, 733.0],
        [826.0, 1123.0],
        [814.0, 901.0],
        [1033.0, 1134.0],
        [1429.0, 1625.0],
        [766.0, 1109.0],
        [624.0, 1240.0],
        [880.0, 970.0],
        [536.0, 1102.0],
        [699.0, 1295.0],
        [674.0, 1066.0],
        [683.0, 1018.0],
        [458.0, 1036.0],
        [1330.0, 1425.0],
        [1162.0, 1130.0],
        [1475.0, 887.0],
        [1224.0, 1596.0],
        [1273.0, 1390.0],
        [972.0, 168.0],
        [1352.0, 1648.0],
        [1149.0, 1244.0],
        [1531.0, 1064.0],
        [419.0, 1114.0],
        [1215.0, 1998.0],
        [494.0, 1060.0],
        [1249.0, 1879.0],
        [1081.0, 492.0],
        [1239.0, 665.0],
        [472.0, 1186.0],
        [1190.0, 1557.0],
        [1160.0, 1031.0],
        [1231.0, 1646.0],
        [164.0, 1243.0],
        [768.0, 1255.0],
        [1205.0, 1275.0],
        [1004.0, 440.0],
        [842.0, 959.0],
        [178.0, 1344.0],
        [1046.0, 413.0],
        [1655.0, 896.0],
        [1245.0, 1696.0],
        [95.0, 1217.0],
        [1461.0, 937.0],
        [748.0, 1150.0],
        [1420.0, 1752.0],
        [1202.0, 1499.0],
        [1259.0, 720.0],
        [1316.0, 1080.0],
        [1031.0, 1238.0],
        [311.0, 1317.0],
        [1121.0, 770.0],
        [1186.0, 1415.0],
        [658.0, 1209.0],
        [1360.0, 493.0],
        [1133.0, 1156.0],
        [1129.0, 1095.0],
        [714.0, 1012.0],
        [953.0, 646.0],
        [1038.0, 369.0],
        [998.0, 179.0],
        [579.0, 1242.0],
        [1199.0, 1471.0],
        [1216.0, 1322.0],
        [1330.0, 1182.0],
        [1258.0, 1355.0],
        [354.0, 1237.0],
        [282.0, 1148.0],
        [1352.0, 1218.0],
        [1357.0, 1060.0],
        [1151.0, 626.0],
    ]
)

targetCoord = np.array(
    [
        [841.305, 1198.7198],
        [927.631, 959.0212],
        [883.99225, 1017.3664],
        [817.9802, 1048.8522],
        [1201.3635, 1341.9391],
        [555.1394, 1098.3268],
        [805.24243, 1166.0364],
        [1101.0247, 1001.0669],
        [722.77527, 1218.2245],
        [1301.3572, 916.526],
        [966.2445, 951.17334],
        [1163.7212, 891.9999],
        [1127.1575, 920.8158],
        [1232.826, 769.2345],
        [877.7586, 1180.9314],
        [1135.7284, 811.8884],
        [1185.2078, 1075.5138],
        [1155.4294, 981.7921],
        [1177.9464, 851.7111],
        [1044.851, 669.8555],
        [997.3276, 1105.8422],
        [889.4511, 1134.4858],
        [758.55566, 1060.8076],
        [615.8414, 1077.3485],
        [1478.8575, 897.7042],
        [1087.7397, 724.7428],
        [835.89655, 1135.4297],
        [796.4341, 917.0598],
        [1037.1779, 1128.1162],
        [1464.5718, 1595.6478],
        [772.7179, 1126.6959],
        [639.43396, 1271.5226],
        [871.88635, 980.8919],
        [541.77, 1141.411],
        [710.08466, 1307.586],
        [673.43414, 1092.8995],
        [680.78253, 1044.3134],
        [457.99792, 1082.3612],
        [1358.5415, 1397.2504],
        [1168.1372, 1117.7571],
        [1450.1407, 851.1212],
        [1266.3652, 1576.3496],
        [1303.3948, 1368.235],
        [893.2751, 173.98154],
        [1391.208, 1619.9385],
        [1163.7201, 1231.0295],
        [1529.5299, 1025.3541],
        [420.80362, 1152.5759],
        [1293.5905, 1975.75],
        [490.41705, 1094.5231],
        [1314.297, 1855.6548],
        [1034.655, 492.906],
        [1199.2773, 645.4932],
        [482.2822, 1227.3713],
        [1225.7625, 1541.3878],
        [1156.5476, 1016.4378],
        [1278.5267, 1625.4044],
        [178.56161, 1306.1583],
        [780.8541, 1268.3405],
        [1222.4808, 1259.4805],
        [964.2754, 441.74496],
        [835.3265, 972.153],
        [204.53769, 1413.2614],
        [991.83563, 411.2698],
        [1650.6266, 834.4475],
        [1298.1508, 1671.5724],
        [108.93118, 1291.5918],
        [1439.2063, 899.58514],
        [756.3671, 1169.9126],
        [1479.6486, 1712.5007],
        [1238.5244, 1480.4994],
        [1224.6327, 695.318],
        [1316.0406, 1055.7354],
        [1046.162, 1234.4912],
        [331.9895, 1367.8849],
        [1095.4545, 761.68115],
        [1215.309, 1397.6898],
        [671.8784, 1237.644],
        [1298.4594, 463.35223],
        [1139.2153, 1145.7471],
        [1132.0254, 1083.1497],
        [710.00037, 1035.7677],
        [917.9624, 651.2084],
        [977.56433, 368.5328],
        [924.2994, 182.4247],
        [590.2328, 1265.0858],
        [1234.2955, 1451.9825],
        [1236.1349, 1304.6243],
        [1339.1342, 1154.5645],
        [1281.9696, 1333.1907],
        [371.91956, 1293.5822],
        [291.38516, 1206.5095],
        [1363.0586, 1191.5559],
        [1355.1196, 1031.3917],
        [1111.2416, 614.707],
    ]
)
dx,dy,dth = 0.438,0.096,np.deg2rad(5.212)
R_gt = np.array([
    [np.cos(dth),-np.sin(dth)],
    [np.sin(dth),np.cos(dth)],
])

srcCoord[:,0] -= 1012
srcCoord[:,1] -= 1012
targetCoord[:,0] -= 1012
targetCoord[:,1] -= 1012

h_gt = np.array([[dx],[dy]]) / RANGE_RESOLUTION_CART_M

R_fit, h_fit = calculateTransformDxDth(srcCoord, targetCoord)

theta_fit = np.arctan2(R_fit[1, 0], R_fit[0, 0]) * 180 / np.pi
print(f"Fitted Transform:\ntheta:\n{theta_fit}\nh:\n{h_fit * RANGE_RESOLUTION_CART_M}")

# yikers
# Visualize
# srcCoord2 = (R_fit @ targetCoord.T + h_fit).T
# srcCoord3 = (R_gt @ targetCoord.T + h_gt).T
# plotFakeFeatures(srcCoord3, targetCoord, srcCoord2,
#                     title_append="", alpha=0.5, clear=False, show=True)
from pulp import *
import time
import pandas as pd
import plotly as py
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
scenario_name = 'scenario_1'  # <<<----- 名称可以修改
# 更改要选择的仓库数量。这个数字应该是一个整数
number_of_whs = 4
number_of_farms = 3
# 可以改变距离带。波段数字表示到的距离。这些数字应该是增加的。
distance_band_1 = 200
distance_band_2 = 400
distance_band_3 = 800
distance_band_4 = 1600
# 如果您不想绘制输入或输出映射，请键入大写F的False来代替True
input_map = True  # <<<----- you can change these from True to False
output_map = True
# 每千克货物的单位公里运价
ob_trans_cost = 5  # 单位条件下的运出运输成本
ic_trans_cost = 5  # 单位条件下，公司间运输成本
ob_min_trans = 20  # 出境最低每千克收费
ic_min_trans = 20  # 公司间运输每千克最低收费
ob_rot_cost = 5.5  # 单位变质成本
pc_p_cost = 0.75  # 工厂的单位处理成本
pc_u_cost = 2.6  # 回收中心的单位处理成本
max_transport_capacity1 = 8820  # 工厂的运输上限
max_transport_capacity2 = 8820  # 配送中心的运输上限
max1 = 10000  # 回收处理中心的处理上线额
max2 = 20000  # 农场货物存储上限
max3 = 20000  # 工厂货物存储上限
a = 8  # 农场每千克货物的售价
# {('工厂名称', '产品名称'): 容量）  每一行是一个工厂。 你可以更改括号中两个数字后面的数字。
# 当一个工厂可以生产一种产品时，该工厂有一个生产上限，比如210000000。
# 例如，“（1,2）：”后第一行的数字是工厂1可以生产的产品2的数量。
plant_product_info = {(1, 1): 21000}
# 这意味着你可以强迫它打开或关闭。你可以更改名字后面的两个数字。
# 0，1来让解算器拾取；1，1来强制使用它；或者0，0来强制关闭它。1,0是一个错误-不要使用它

wh_status = {1: ('仓库1', 0, 1),
             2: ('仓库2', 0, 1),
             3: ('仓库3', 0, 1),
             4: ('仓库4', 0, 1)}

farm_status = {1: ('农场1', 0, 1),
               2: ('农场2', 0, 1),
               3: ('农场3', 0, 1)}


# 设置输入数据
def get_data():
    # {'客户': ('所在城市名称', '状态', '城市', '城市所在州', '邮政代码', '城市所在国家或地区', '纬度', '经度')}
    customers = {1: ('聊城', True, 'LC', '山东', 252000, '中国', 36.264851, 115.716208),
                 2: ('潍坊', True, 'WF', '山东', 261000, '中国', 36.62285, 118.544436),
                 3: ('济宁', True, 'JN', '山东', 272100, '中国', 35.741894, 116.509156),
                 4: ('德州', True, 'DZ', '山东', 253000, '中国', 37.297537, 116.291132)}

    # {'仓库': ('所在城市名称', '状态', 'Status', '仓库所在城市', '城市所在州', '邮政编码', '城市所在国家或地区', '纬度', '经度')}
    warehouses = {1: ('济南', False, '可用', 'JN', '山东', 250000, '中国', 36.695386, 116.947544),
                  2: ('菏泽', False, '可用', 'HZ', '山东', 274000, '中国', 35.535201, 115.931994),
                  3: ('聊城', False, '可用', 'LC', '山东', 252000, '中国', 36.511414, 115.999142),
                  4: ('淄博', False, '可用', 'ZB', '山东', 255000, '中国', 36.197327, 118.196222)}

    # {'工厂': ('工厂名称', '工厂所在城市', '工厂所在州', '无邮政编码', '城市所在国家或地区', '纬度', '经度')}
    plants = {1: ('工厂1', '菏泽', '山东', 'nan', '中国', 34.714363, 115.816943)}

    # {'农场': ('农场名称', '状态', 'Status', '仓库所在城市','仓库所在省份'}
    farms = {1: ('曹县一场', True, '可用', '菏泽', '山东'),
             2: ('曹县二场', True, '可用', '菏泽', '山东'),
             3: ('曹县三场', True, '可用', '菏泽', '山东')}

    # {'产品': ('名称')}
    products = {1: '产品1'}

    # {'回收中心': ('回收中心名称', '回收中心所在城市', '回收中心所在州', '无邮政编码', '回收中心所在国家或地区', '纬度', '经度')}
    recycles = {1: ('回收中心', '菏泽', '山东', 'nan', '中国', 34.864858, 115.248142)}

    # {('客户id', '产品id'): '需求数量/吨'}  （(1, 1): 32007.5）表示1号客户需要32007.5吨的产品1
    customer_demands = {(1, 1): 850, (2, 1): 970.6, (3, 1): 679.8, (4, 1): 986.1}

    # {('工厂ID', '产品ID'): 容量)}
    plant_product_info = {(1, 1): 210000000}

    #  {('农场ID’,‘工厂id'): '距离'}
    farm_plant_distance = {(1, 1): 28.1, (2, 1): 30.1, (3, 1): 60.3}

    # {('工厂ID’,‘仓库id'): '距离'}
    plant_wh_distance = {(1, 1): 330.5, (1, 2): 120.5, (1, 3): 239.7, (1, 4): 395.1}

    # {('仓库id', '客户ID'): ‘距离’'}
    wh_cust_distance = {(1, 1): 140.9, (1, 2): 106.6, (1, 3): 154.7, (1, 4): 176.7,
                        (2, 1): 123.9, (2, 2): 369.4, (2, 3): 97.4, (2, 4): 280.6,
                        (3, 1): 42.6, (3, 2): 269.8, (3, 3): 142.3, (3, 4): 135.6,
                        (4, 1): 263.9, (4, 2): 75.2, (4, 3): 214.6, (4, 4): 255.1}
    # {('仓库id', '客户ID'): ‘变质率’'}
    wh_cus_rot = {(1, 1): 0.000173, (1, 2): 0.000131, (1, 3): 0.000190, (1, 4): 0.000217,
                  (2, 1): 0.000152, (2, 2): 0.000454, (2, 3): 0.000119, (2, 4): 0.000345,
                  (3, 1): 0.0000524, (3, 2): 0.000332, (3, 3): 0.000175, (3, 4): 0.00016,
                  (4, 1): 0.000324, (4, 2): 0.0000925, (4, 3): 0.000264, (4, 4): 0.000314}
    # {('客户ID','回收中心'): ‘运输距离’'}
    cus_u_distance = {(1, 1): 191.8, (2, 1): 487.5, (3, 1): 181.4, (4, 1): 348.1}
    #
    return customers, warehouses, plants, farms, products, recycles, customer_demands, farm_plant_distance, plant_wh_distance, wh_cust_distance, wh_cus_rot, cus_u_distance


def optimal_location(number_of_whs, farms, warehouses, customers, plants, products, recycles, plant_product_info,
                     customer_demands,
                     distance_band, scenario_name):
    start_time = time.time()

    # 问题初始化
    jade_problem = LpProblem("JADE", LpMinimize)

    # 增加从农场到工厂的运输决策变量
    flow_vars_fp = LpVariable.dicts("flow_fp", [(f, p, k) for p in plants for f in farms for k in products
                                                if (p, k) in plant_product_info.keys()], lowBound=0, cat='Continuous')
    # 增加从工厂到仓库的运输决策变量
    flow_vars_pw = LpVariable.dicts("flow_pw", [(p, w, k) for p in plants for w in warehouses for k in products
                                                if (p, k) in plant_product_info.keys()], lowBound=0, cat='Continuous')
    # 添加从仓库到客户的运输决策变量
    flow_vars_wc = LpVariable.dicts("flow_wc", [(w, c, k) for w in warehouses for c in customers for k in products if
                                                customer_demands[c, k] > 0],
                                    lowBound=0, upBound=1, cat='Integer')
    # 根据仓库已打开或未打开，来添加设施的0-1变量
    facility_vars = LpVariable.dicts("facility_vars", [w for w in warehouses], lowBound=0, upBound=1, cat='Integer')

    # 根据农场已打开或未打开，来添加设施的0-1变量
    facility_vars_farm = LpVariable.dicts("facility_vars_farm", [f for f in farms], lowBound=0, upBound=1,
                                          cat='Integer')

    # 为仓库向客户添加单一来源约束。单仓库对应单客户
    single_source_wc = LpVariable.dicts("single_source_wc", [(w, c) for w in warehouses for c in customers],
                                        lowBound=0, upBound=1, cat='Integer')
    # 为农场向工厂添加单一来源约束。单农场对应单工厂
    single_source_fp = LpVariable.dicts("single_source_fp", [(f, p) for f in farms for p in plants],
                                        lowBound=0, upBound=1, cat='Integer')

    # 约束
    # 每个客户和每种产品都必须有至少一个仓库为其服务，不会出现某个客户的需求无法被任何仓库服务的情况。约束2.1
    for c in customers:
        for k in products:
            if customer_demands[c, k] > 0:
                jade_problem += LpConstraint(e=lpSum([flow_vars_wc[w, c, k] for w in warehouses]),
                                             sense=LpConstraintEQ,
                                             name=str(c) + '_' + str(k) + "_Served",
                                             rhs=1)
    # 要求所有仓库向客户的运输量之和不大于所有客户的总需求量。约束2.2
    jade_problem += LpConstraint(e=lpSum([flow_vars_wc[w, c, k] for w in warehouses]),
                                 sense=LpConstraintLE,
                                 name=str(c) + '_' + str(k) + "_Capacity",
                                 rhs=sum([customer_demands[c, k] for c in customers for k in products if
                                          customer_demands[c, k] > 0]))

    # 从仓库流出的产品总量必须等于从工厂流入该仓库的产品总量，确保所有从工厂流出的产品都能被送到客户手中(约束2.3)
    for w in warehouses:
        for k in products:
            jade_problem += LpConstraint(e=lpSum([flow_vars_wc[w, c, k] * customer_demands[c, k] for c in customers if
                                                  customer_demands[c, k] > 0]) - lpSum(
                [flow_vars_pw[p, w, k] for p in plants if (p, k) in plant_product_info.keys()]),
                                         sense=LpConstraintEQ,
                                         name=str(w) + '_' + str(k) + "_FlowConstraint",
                                         rhs=0)

    # 农场运输到工厂的货物数量不超过工厂的存储容量上限，约束2.4,2.6
    for f in farms:
        jade_problem += LpConstraint(e=lpSum([flow_vars_fp[f, p, k] for p in plants]),
                                     sense=LpConstraintLE,
                                     name=str(f) + "_FarmCapacity",
                                     rhs=max3)

    #  农场的运输货物不超过农场的最大容量，约束2.5
    for p in plants:
        for w in warehouses:
            jade_problem += flow_vars_pw[p, w, k] <= max2

    # 仓库容量约束。对于每个工厂和产品的组合，所有从该工厂流向仓库的该产品的流量之和不能超过该工厂的生产能力；约束2.7
    for (p, k), value in plant_product_info.items():
        jade_problem += LpConstraint(e=lpSum([flow_vars_pw[p, w, k] for w in warehouses]),
                                     sense=LpConstraintLE,
                                     name=str(p) + '_' + str(k) + "_plantproductConstraint",
                                     rhs=plant_product_info[p, k])
    # 运输数量不超过车辆运输数量，约束2.8
    for p in plants:
        jade_problem += LpConstraint(e=lpSum([flow_vars_pw[p, w, k] for w in warehouses]),
                                     sense=LpConstraintLE,
                                     name=str(p) + "_PlantTransportCapacity",
                                     rhs=max_transport_capacity1)
    for w in warehouses:
        for k in products:
            jade_problem += LpConstraint(
                e=lpSum([flow_vars_wc[w, c, k] for c in customers if customer_demands[c, k] > 0]),
                sense=LpConstraintLE,
                name=str(w) + '_' + str(k) + "_WarehouseTransportCapacity",
                rhs=max_transport_capacity2)
    # 保证要处理的变质产品不超过处理中心的处理限额。约束2.9
    for w in warehouses:
        for c in customers:
            jade_problem += LpConstraint(e=flow_vars_wc[w, c, k] * wh_cus_rot[w, c],
                                         sense=LpConstraintLE,
                                         name=str(w) + '_' + str(c) + '_max1_constraint',
                                         rhs=max1)

    # 如果有来自仓库的流量，则仓库应设置为打开，即1。Yjrt  0-1变量
    for w in warehouses:
        jade_problem += LpConstraint(
            e=lpSum([flow_vars_wc[w, c, k] for c in customers for k in products if customer_demands[c, k] > 0]) -
              facility_vars[w] * 10000000,
            sense=LpConstraintLE,
            name=str(w) + "_facilityopenconstraint",
            rhs=0)
    # 如果有来自农场的流量，则农场应设置为打开，即1。Yfpt  0-1变量
    for f in farms:
        jade_problem += LpConstraint(
            e=lpSum([flow_vars_fp[f, p, k] for p in plants for k in products if (p, k) in plant_product_info and
                     all(flow_vars_pw[p, w, k].value() is not None and flow_vars_pw[p, w, k].value() >= 0 for w in
                         warehouses)
                     ]) - facility_vars_farm[f] * 10000000,
            sense=LpConstraintLE,
            name=str(f) + "_farmopenconstraint",
            rhs=0)

    # 限制开放仓库的数量
    jade_problem += LpConstraint(e=lpSum([facility_vars[w] for w in warehouses]),
                                 sense=LpConstraintEQ,
                                 name="numberofwhsconstraints",
                                 rhs=number_of_whs)
    # 限制开放农场数量
    jade_problem += LpConstraint(e=lpSum([facility_vars_farm[f] for f in farms]),
                                 sense=LpConstraintEQ,
                                 name="numberoffarmsconstraints",
                                 rhs=number_of_farms)

    # 使用用户的状态修复解决方案中的一些仓库
    for facility in warehouses:
        w = facility
        jade_problem += LpConstraint(e=facility_vars[facility],
                                     sense=LpConstraintGE,
                                     name=str(facility) + "_" + "Lower Bound",
                                     rhs=wh_status[w][1])
    for facility in warehouses:
        w = facility
        jade_problem += LpConstraint(e=facility_vars[facility],
                                     sense=LpConstraintLE,
                                     name=str(facility) + "_" + "Upper Bound",
                                     rhs=wh_status[w][2])

    # 要求农场在开放时，值为1，关闭时值为0
    for farm in farms:
        f = farm
        jade_problem += LpConstraint(e=facility_vars_farm[farm],
                                     sense=LpConstraintGE,
                                     name=str(farm) + "_" + "Lower Bound 1",
                                     rhs=farm_status[f][1])
    for farm in farms:
        f = farm
        jade_problem += LpConstraint(e=facility_vars_farm[farm],
                                     sense=LpConstraintLE,
                                     name=str(farm) + "_" + "Upper Bound 1",
                                     rhs=farm_status[f][2])

    # 分两步设置单个约束。第一，先将单个源与流量变量联系起来。约束2.1
    for w in warehouses:
        for c in customers:
            for k in products:
                if customer_demands[c, k] > 0:
                    jade_problem += LpConstraint(e=flow_vars_wc[w, c, k] - single_source_wc[w, c],
                                                 sense=LpConstraintLE,
                                                 name=str(w) + '_' + str(c) + '_' + str(k) + "_Single Source Check",
                                                 rhs=0)

    # 其次，我们需要防止两个仓库为同一客户提供服务。约束2.1
    for c in customers:
        jade_problem += LpConstraint(e=lpSum([single_source_wc[w, c] for w in warehouses]),
                                     sense=LpConstraintLE,
                                     name=str(c) + "_Single Source Constraint",
                                     rhs=1)

    # 非负约束
    for f in farms:
        for p in plants:
            jade_problem += flow_vars_fp[f, p, k] >= 0
    for p in plants:
        for w in warehouses:
            jade_problem += flow_vars_pw[p, w, k] >= 0
    for c in customers:
        for k in products:
            if customer_demands[c, k] > 0:
                jade_problem += LpConstraint(e=lpSum([flow_vars_wc[w, c, k] for w in warehouses]),
                                             sense=LpConstraintGE,
                                             name=str(c) + '_' + str(k) + "_NonNegative",
                                             rhs=0)

    # 创建入站和出站成本以计算最低成本值
    f_cost = {}  # 订货成本
    for p in plants:
        for f in farms:
            f_cost[f, p] = float(a)

    f_p_cost = {}  # 农场工厂运输的单位运输成本
    for p in plants:
        for f in farms:
            f_p_cost[f, p] = ic_trans_cost * farm_plant_diatance[f, p]
            if farm_plant_diatance[f, p] <= ic_min_trans / ic_trans_cost:
                f_p_cost[f, p] = ic_min_trans
    p_w_cost = {}  # 工厂向配送中心运输的单位运输成本
    for p in plants:
        for w in warehouses:
            p_w_cost[p, w] = ic_trans_cost * plant_wh_distance[p, w]
            if plant_wh_distance[p, w] <= ic_min_trans / ic_trans_cost:
                p_w_cost[p, w] = ic_min_trans

    w_c_cost = {}  # 配送中心向客户运输的单位运输成本
    for w in warehouses:
        for c in customers:
            w_c_cost[w, c] = ob_trans_cost * wh_cust_distance[w, c]
            if wh_cust_distance[w, c] <= ob_min_trans / ob_trans_cost:  # this division is now in miles units
                w_c_cost[w, c] = ob_min_trans
    f_p_cost = {}  # 农场的订货成本
    for p in plants:
        for f in farms:
            f_p_cost[f, p] = ic_trans_cost * farm_plant_diatance[f, p]
            if farm_plant_diatance[f, p] <= ic_min_trans / ic_trans_cost:
                f_p_cost[f, p] = ic_min_trans

    w_c_rot_cost = {}  # 配送中心向客户运输的单位变质成本
    for w in warehouses:
        for c in customers:
            w_c_rot_cost[w, c] = ob_rot_cost * wh_cus_rot[w, c]
    c_u_cost = {}  # 客户向回收中心运输的单位运输成本
    for c in customers:
        for u in recycles:
            c_u_cost[c, u] = ob_trans_cost * cus_u_distance[c, u]
    for p in plants:
        for f in farms:
            for k in products:
                if (p, k) in plant_product_info.keys():
                    print("Type of f_cost[f, p]: ", type(f_cost[f, p]))
                    print("Value of f_cost[f, p]: ", f_cost[f, p])

    # 设置目标函数
    # total_weighted_demand_objective = lpSum([flow_vars_pw[p, w, k]*ic_trans_cost*plant_wh_distance[p,w] for p in plants for w in warehouses for k in products if (p,k) in plant_product_info.keys()]) + lpSum([flow_vars_wc[w, c, k]*wh_cust_distance[w, c]*customer_demands[c, k]*ob_trans_cost for w in warehouses for c in customers for k in products])
    # 工厂向农场订货的订货成本+农场到工厂的运输成本+工厂到配送中心的运费+配送中心到客户的运费+配送中心到客户的变质成本+退回货物的运输成本
    v = 70  # 车辆平均速度
    total_weighted_demand_objective = lpSum([
        flow_vars_fp[f, p, k] * f_cost[f,p] for p in plants for f in farms for k in products if
        (p, k) in plant_product_info.keys()]) + lpSum(
        [flow_vars_fp[f, p, k] * f_p_cost[f, p] * 1 / v for p in plants for f in farms for k in products if
         (p, k) in plant_product_info.keys()]) + lpSum(
        [flow_vars_pw[p, w, k] * p_w_cost[p, w] * 1 / v for p in plants for w in warehouses for k in products if
         (p, k) in plant_product_info.keys()]) + lpSum(
        [flow_vars_wc[w, c, k] * w_c_cost[w, c] * customer_demands[c, k] * 1 / v for w in warehouses for c in customers
         for k in
         products]) + lpSum(
        [flow_vars_wc[w, c, k] * w_c_rot_cost[w, c] * customer_demands[c, k] for w in warehouses for c in customers for
         k in products]) + lpSum(
        [flow_vars_wc[w, c, k] * c_u_cost[c, u] * customer_demands[c, k] * 1 / v for w in warehouses for c in customers
         for k in products for u in recycles])

    jade_problem.setObjective(total_weighted_demand_objective)

    jade_problem.solve()

    total_flow_fp = {(f, p, k): flow_vars_fp[f, p, k].varValue for p in plants for f in farms for k in products if
                     (p, k) in plant_product_info.keys() and flow_vars_fp[f, p, k].varValue > 0}

    total_flow_pw = {(p, w, k): flow_vars_pw[p, w, k].varValue for p in plants for w in warehouses for k in products if
                     (p, k) in plant_product_info.keys() and flow_vars_pw[p, w, k].varValue > 0}

    total_flow_wc = {(w, c, k): flow_vars_wc[w, c, k].varValue for w in warehouses for c in customers for k in products
                     if flow_vars_wc[w, c, k].varValue > 0}

    print('Status:' + LpStatus[jade_problem.status])
    print("Objective: " + str(jade_problem.objective.value()))
    file.write('\nStatus:' + LpStatus[jade_problem.status])

    total_demand = sum(customer_demands.values())
    file.write("\nTotal Demand:" + str(total_demand))

    file.write("\nObjective: " + str(jade_problem.objective.value()))

    end_time = time.time()

    time_diff = end_time - start_time

    file.write("\nRun Time in seconds {:.1f}".format(time_diff))
    print("Run Time in seconds {:.1f}".format(time_diff))

    # 准备数据写入excel表格
    opened_warehouses = []
    warehouse_list = []

    opened_farms = []
    farms_list = []

    for w in facility_vars.keys():
        if facility_vars[w].varValue > 0:
            warehouse_list.append(w)

    for f in facility_vars_farm.keys():
        if facility_vars_farm[f].varValue > 0:
            farms_list.append(f)
    list_warehouses_open = set(list(warehouse_list))
    total_demand_to_warehouse = {w: sum(customer_demands[c, k] * flow_vars_wc[w, c, k].varValue for c in customers)
                                 for w in list_warehouses_open for k in products}

    list_farms_open = set(list(farms_list))
    total_demand_to_farm = {f: sum(flow_vars_fp[f, p, k].varValue for p in plants)
                            for f in list_farms_open for k in products}

    for w in list_warehouses_open:
        wh = {
            '配送中心ID': w,
            '配送中心所在城市': warehouses[w][0],
            '省份': warehouses[w][4],
            '邮政编码': warehouses[w][5],
            '纬度': warehouses[w][7],
            '经度': warehouses[w][8],
            '配送中心的总需求': total_demand_to_warehouse[w]
        }
        opened_warehouses.append(wh)

    for f in list_farms_open:
        farm = {
            '农场': f,
            '农场名称': farms[f][0],
            '状态': farms[f][1],
            '农场所在城市': farms[f][2],
            '农场所在省份': farms[f][3],
            '农场总运输量': total_demand_to_farm[f]

        }
        opened_farms.append(farm)
    df_wh = pd.DataFrame.from_records(opened_warehouses)
    df_wh = df_wh[['配送中心ID', '配送中心所在城市', '省份', '邮政编码', '配送中心的总需求']]
    df_fa = pd.DataFrame.from_records(opened_farms)
    df_fa = df_fa[['农场', '农场名称', '状态', '农场所在城市', '农场所在省份', '农场总运输量']]
    # writing detailed files
    writer = pd.ExcelWriter(r"D:\桌面文件夹哦\四级供应链模型\模型求解\农场测试.xlsx")
    df_wh.to_excel(writer, '开放的配送中心', index=False)
    df_fa.to_excel(writer, '农场情况', index=False)

    customers_assignment = []
    maps_assign_wc = {}
    for (w, c, k) in total_flow_wc.keys():
        cust = {
            '配送中心': str(warehouses[w][0] + ',' + warehouses[w][4]),
            '客户': str(customers[c][0] + ',' + customers[c][3]),
            '产品': str(k),
            '客户需求': customer_demands[c, k],
            '距离': wh_cust_distance[w, c],
            '配送中心纬度': warehouses[w][7],
            '配送中心经度': warehouses[w][8],
            '客户纬度': customers[c][6],
            '客户经度': customers[c][7]
        }
        customers_assignment.append(cust)
        if (w, c) not in maps_assign_wc:
            maps_assign_wc[w, c] = 1

    plants_assignment = []
    maps_assign_pw = {}
    for (p, w, k) in total_flow_pw.keys():
        plant = {
            '配送中心': str(warehouses[w][0] + ',' + warehouses[w][4]),
            '工厂': str(plants[p][1] + ',' + plants[p][2]),
            '产品': str(k),
            '数量': total_flow_pw[p, w, k],
            '距离': plant_wh_distance[p, w],
            '配送中心纬度': warehouses[w][7],
            '配送中心经度': warehouses[w][8],
            '工厂纬度': plants[p][5],
            '工厂经度': plants[p][6]
        }
        plants_assignment.append(plant)
        if (p, w) not in maps_assign_pw:
            maps_assign_pw[p, w] = 1

    df_cu = pd.DataFrame.from_records(customers_assignment)
    df_cu_copy = df_cu.copy()
    df_cu = df_cu[['配送中心', '客户', '产品', '距离', '客户需求']]
    df_cu.to_excel(writer, '配送中心到客户分配情况', index=False)

    df_pl = pd.DataFrame.from_records(plants_assignment)
    df_pl_copy = df_pl.copy()
    df_pl = df_pl[['工厂', '配送中心', '产品', '距离', '数量']]
    df_pl.to_excel(writer, '工厂到配送中心分配情况', index=False)

    writer.close()

    distance_band_1 = distance_band[0]
    distance_band_2 = distance_band[1]
    distance_band_3 = distance_band[2]
    distance_band_4 = distance_band[3]
    # 写入每个距离点内的需求百分比
    total_demand = sum(df_cu['客户需求'])
    percent_demand_distance_band_1 = sum(
        df_cu[df_cu['距离'] < distance_band_1]['客户需求']) * 100 / total_demand
    percent_demand_distance_band_2 = sum(
        df_cu[df_cu['距离'] < distance_band_2]['客户需求']) * 100 / total_demand
    percent_demand_distance_band_3 = sum(
        df_cu[df_cu['距离'] < distance_band_3]['客户需求']) * 100 / total_demand
    percent_demand_distance_band_4 = sum(
        df_cu[df_cu['距离'] < distance_band_4]['客户需求']) * 100 / total_demand
    file.write(
        "\n {} 公里内需求完成百分比 : {:.1f}".format(distance_band[0], percent_demand_distance_band_1))
    file.write(
        "\n {} 公里内需求完成百分比 : {:.1f}".format(distance_band[1], percent_demand_distance_band_2))
    file.write(
        "\n {} 公里内需求完成百分比 : {:.1f}".format(distance_band[2], percent_demand_distance_band_3))
    file.write(
        "\n {} 公里内需求完成百分比 : {:.1f}".format(distance_band[3], percent_demand_distance_band_4))

    return df_pl_copy, df_cu_copy, df_fa, list_warehouses_open


def test_input(farms, warehouses, customers, plants, recycles, products, customer_demands, farm_plant_diatance,
               plant_wh_distance,
               wh_cust_distance,
               wh_cust_rot, cus_u_distance,
               distance_band, number_of_whs, plant_product_info):
    for f in farms:
        for p in plants:
            if (farm_plant_diatance[f, p] >= 0):
                pass
            else:
                file.write(f'\n农场{f} 和工厂 {p} 不可用或者不存在')
    for c in customers:
        for w in warehouses:
            if (wh_cust_distance[w, c] >= 0):
                pass
            else:
                file.write(f'\n配送中心 {w} 和客户 {c} 不可用或者不存在')

    for p in plants:
        for w in warehouses:
            if (plant_wh_distance[p, w] >= 0):
                pass
            else:
                file.write(f'\n工厂 {p} 和配送中心 {w} 不可用或者不存在')
    for u in recycles:
        for c in customers:
            if (cus_u_distance[c, u] >= 0):
                pass
            else:
                file.write(f'\n 客户 {c} 和回收中心 {u} 不可用或者不存在')
    for w in warehouses:
        for c in customers:
            if (wh_cust_rot[w, c] >= 0):
                pass
            else:
                file.write(f'\n配送中心{w} 和客户 {c} 之间的变质率不符合要求')

    for k in products:
        for c in customers:
            if (customer_demands[c, k] >= 0):
                pass
            else:
                file.write(f'\n客户 {c} 的需求无法满足')

    if ((distance_band[0] < distance_band[1]) and (distance_band[1] < distance_band[2]) and (
            distance_band[2] < distance_band[3])):
        pass
    else:
        file.write(f'\n距离圈不按照递增序列存在')

    if isinstance(number_of_whs, int) == False:
        file.write(f'\n配送中心的数目 = {number_of_whs} 不是整数')
    # 约束2.2
    for k in products:
        demand = sum(customer_demands[c, k] for c in customers if customer_demands[c, k] > 0)
        capacity = sum(plant_product_info[p, k] for p in plants if (p, k) in plant_product_info.keys())
        if demand > capacity:
            file.write(f'\n产品 {k} 的需求量高于存储量')


distance_band = [distance_band_1, distance_band_2, distance_band_3, distance_band_4]

print("运行时间可能会耗费10~30秒.")
file = open(scenario_name + '总结文档' + '.txt', "w")

file.write('获得输入数据')
customers, warehouses, plants, farms, products, recycles, customer_demands, farm_plant_diatance, plant_wh_distance, wh_cust_distance, wh_cus_rot, cus_u_distance = get_data()

for k in products:
    a = ''
    for p in plants:
        if (p, k) in plant_product_info.keys():
            if plant_product_info[p, k] > 0:
                a = a + (str(p) + " ")
    a = a.replace(" ", ",")
    file.write('\n产品 ' + str(k) + ' 可以在工厂中生产 ' + a[:-1])

file.write('\nT测试的输入数据')
test_input(farms, warehouses, customers, plants, recycles, products, customer_demands, farm_plant_diatance,
           plant_wh_distance, wh_cust_distance,
           wh_cus_rot, cus_u_distance, distance_band, number_of_whs, plant_product_info)

file.write('\n建立模型')
df_pl, df_cu, df_fa, wh_loc = optimal_location(number_of_whs, farms, warehouses, customers, plants, products, recycles,
                                               plant_product_info,
                                               customer_demands, distance_band, scenario_name)
file.close()

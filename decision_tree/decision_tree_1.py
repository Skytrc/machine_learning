"""
https://zhuanlan.zhihu.com/p/20794583
"""
from math import log


def uniquecounts(rows):
    """
    对y的各种可能取值出现的个数进行计算。其他函数利用该函数数据集和
    的混杂程度
    :param rows:数据集
    :return: 统计结果返回的数据集
    """
    results = {}
    for row in rows:
        # 计算结果在最后一行
        r = row[len(row)-1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def entropy(rows):
    """
    计算results的熵
    :param rows:数据集
    :return:数据集各个值的熵
    """
    log_ent = lambda x: log(x)/log(2)
    results = uniquecounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]/len(rows))
        ent = ent - p*log_ent(p)
    return ent


class decisionnode:
    """
    定义节点属性
    """
    def __init__(self, col=-1, value=None, results=None,tb=None, fb=None):
        # col是待检验的判断条件所对应的列索引值
        self.col = col
        # value对应于为了使结果为True，当前列必须匹配的值
        self.value = value
        # 保存的是针对当前分支的结果，它是一个字典
        self.results = results
        # desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.tb = tb
        # desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb


def giniimpurity(rows):
    """
    判断树纯度的gini值
    """
    length = len(rows)
    counts = uniquecounts(rows)
    imp = 0.0
    for k in counts:
        imp += (counts[k] / length) ** 2
    return 1 - imp


def divideset(rows, column, value):
    """
    在某一列上对数据集进行拆分。可英语于数值型或因子型变量
    定义一个函数，判断当前数据属于第一组还是第二组
    :param rows: 数据
    :param column: 特征切分
    :param value: 切分标准
    :return: 切分数据
    """
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    # 把数据集拆分成两个集合
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def buildtree(rows, scoref=entropy):
    # ?
    if len(rows) == 0:
        return decisionnode()
    current_score = scoref(rows)

    # 定义一些变量以记录最佳拆分条件
    best_gain = 0.0
    best_criteria = None
    best_set = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # 在当前列中生成一个由不同值构成的序列
        column_value = {}
        for row in rows:
            # 初始化
            column_value[row[col]] = 1
        for value in column_value.keys():
            (set1, set2) = divideset(rows, col, value)

            # 信息增益
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1)>0 and len(set2)>0:
                best_gain = gain
                best_criteria = (col, value)
                best_set = (set1, set2)

    if best_gain > 0:
        true_branch = buildtree(best_set[0])
        false_branch = buildtree(best_set[1])
        return decisionnode(col=best_criteria[0], value=best_criteria[1],
                            tb=true_branch, fb=false_branch)
    else:
        return decisionnode(results=uniquecounts(rows))


def printtree(tree, indent=''):
    # 是否为叶节点
    if tree.results != None:
        print(tree.results)
    else:
        # 打印判断条件
        print(str(tree.col) + ":" + str(tree.value) + "?")
        # 打印分支
        print(indent + "T->"), printtree(tree.tb, indent + " ")
        print(indent + "F->"), printtree(tree.fb, indent + " ")


def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


# test


my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]


divideset(my_data,2,'yes')


giniimpurity(my_data)


tree = buildtree(my_data)


printtree(tree=tree)


classify(['(direct)', 'USA', 'yes', 5], tree)
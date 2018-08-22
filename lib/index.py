def getData2Index(list_item=[]):
    item2index = {}
    indx2item = {}

    for item in list_item:
        item2index[item] = len(item2index)
        indx2item[len(indx2item)] = item

    return item2index, indx2item


if __name__ == '__main__':
    getData2Index(['abc', 'bcd'])

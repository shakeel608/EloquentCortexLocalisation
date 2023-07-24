import csv

def list_of_dicts_to_csv(data_list, filename):
    if not data_list:
        return

    keys = data_list[0].keys()

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_list)


#data_list = [
#    {'name': 'John', 'age': 25, 'city': 'New York'},
#    {'name': 'Emma', 'age': 30, 'city': 'London'},
#    {'name': 'Michael', 'age': 35, 'city': 'Paris'}
#]
#
#list_of_dicts_to_csv(data_list, 'test_files.csv')

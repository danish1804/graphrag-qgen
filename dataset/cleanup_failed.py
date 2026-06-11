import openpyxl

path = 'dataset/outputs/P000233DS_Question_Dataset.xlsx'
wb   = openpyxl.load_workbook(path)
ws   = wb.active

rows_to_delete = []
for row in ws.iter_rows(min_row=3):
    status_cell = row[19]
    id_cell     = row[0]
    if status_cell.value and 'FAILED' in str(status_cell.value):
        rows_to_delete.append(row[0].row)
        print(f'Marking for delete: {id_cell.value}')

for row_num in sorted(rows_to_delete, reverse=True):
    ws.delete_rows(row_num)
    print(f'Deleted row {row_num}')

wb.save(path)
print(f'Done — deleted {len(rows_to_delete)} failed rows')



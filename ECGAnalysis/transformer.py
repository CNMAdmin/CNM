import openpyxl
import os


path = 'F:/SNU/test/'
files = os.listdir(path)

for file in files:
  wb = openpyxl.load_workbook(path+file)
  sheets = wb.get_sheet_names()
  for sheet in sheets:
    cur_sheet = wb.get_sheet_by_name(sheet)
    for i in range(len(list(cur_sheet.columns))):
      for j in range(8, len(list(cur_sheet.rows))):
        print(sur_sheet.cell(colum=i, row=j))


        
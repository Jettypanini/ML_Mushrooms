import pandas as pd
file_name = "encodedShroomsV2.csv"
file_name_output = "encodedShroomsV2Unique.csv"

df = pd.read_csv(file_name, sep=",")
Dup_Rows = df[df.duplicated()]

DF_RM_DUP = Dup_Rows.drop_duplicates(keep='first')

# Notes:
# - the `subset=None` means that every column is used 
#    to determine if two rows are different; to change that specify
#    the columns as an array
# - the `inplace=True` means that the data structure is changed and
#   the duplicate rows are gone  


# Write the results to a different file
DF_RM_DUP.to_csv(file_name_output, index=False)
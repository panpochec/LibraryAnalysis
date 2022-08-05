import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFont, ImageDraw
import os

def sub_library_compiler(column_sub_A, column_sub_B, column_val_A, column_val_B):
    """Function preparing data for heatmap generation

    Parameters
    ----------
    column_sub_A : str
        name of a column containing descriptors A (ring A substitution)
    column_sub_B : str
        name of a column containing descriptors B (ring B substitution)
    column_val_A: str
        name of a column containing values A (analyzed parameter for ring A)
    column_val_B: str
        name of a column containing values B (analyzed parameter for ring B)
    Returns
    -------
    DataFrame
        Returns dataframe containing rectangular data for heatmap.
    """
    # Getting substitution IDs and activation energy for ring A
    library_ring_a = library[[column_sub_A, column_sub_B, column_val_A]]

    library_ring_a.reset_index(drop=True, inplace=True)

    # Setting categorical type for heatmap
    library_ring_a.loc[:, column_sub_A] = library_ring_a.loc[:, column_sub_A].astype('category')
    library_ring_a.loc[:, column_sub_B] = library_ring_a.loc[:, column_sub_B].astype('category')
    library_ring_a.rename(columns={column_sub_A: 'Proximal ring substitution',
                                   column_sub_B: 'Distal ring substitution'},
                          inplace=True)
    # Pivoting table for heatmap
    library_ring_a = library_ring_a.pivot(index='Proximal ring substitution',
                                          columns='Distal ring substitution',
                                          values=column_val_A)

    # Getting substitution IDs and activation energy for ring B
    library_ring_b = library[[column_sub_A, column_sub_B, column_val_B]]
    library_ring_b.loc[:, column_sub_A] = library_ring_b.loc[:, column_sub_A].astype('category')
    library_ring_b.loc[:, column_sub_B] = library_ring_b.loc[:, column_sub_B].astype('category')
    library_ring_b = library_ring_b.pivot(index=column_sub_A, columns=column_sub_B,
                                          values=column_val_B)

    # Appending missing values from the other library (compounds are symmetrical along short axis)
    library_ring_a = library_ring_a.fillna(4)
    for x in range(0, len(library_ring_a)):
        for z in range(0, len(library_ring_a)):
            if library_ring_a.iloc[x, z] == 4:
                library_ring_a.iloc[x, z] = library_ring_b.iloc[z, x]
    library_ring_b = library_ring_b.fillna(4)
    for x in range(0, len(library_ring_b)):
        for z in range(0, len(library_ring_b)):
            if library_ring_b.iloc[x, z] == 4:
                library_ring_b.iloc[x, z] = library_ring_a.iloc[z, x]
    library_ring_a.sort_index(ascending=True, inplace=True)
    library_ring_b.sort_index(ascending=False, inplace=True)
    return library_ring_a


def sub_library_compiler_charges():
    """Function preparing charges table for monosubstituted compounds.
        Returns
        -------
        DataFrame
            Returns dataframe containing charges.
        """
    col0 = []
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    colNR = []
    colO = []
    colN = []

    for x in range(0, len(library)):
        if library.iloc[x, -2] == "None" and library.iloc[x, -1] != "None":  #ring A
            col0.append(library.iloc[x, -1])
            col5.append(library.iloc[x, 22])
            col6.append(library.iloc[x, 23])
            col4.append(library.iloc[x, 24])
            col3.append(library.iloc[x, 25])
            col2.append(library.iloc[x, 26])
            col1.append(library.iloc[x, 27])
            colNR.append(library.iloc[x, 28])
            colN.append(library.iloc[x, 32])
            colO.append(library.iloc[x, 33])
        elif library.iloc[x, -1] == "None" and library.iloc[x, -2] != "None":  # ring B
            col0.append(library.iloc[x, -2])
            col5.append(library.iloc[x, 14])
            col6.append(library.iloc[x, 15])
            col4.append(library.iloc[x, 16])
            col3.append(library.iloc[x, 17])
            col2.append(library.iloc[x, 18])
            col1.append(library.iloc[x, 19])
            colNR.append(library.iloc[x, 20])
            colN.append(library.iloc[x, 30])
            colO.append(library.iloc[x, 31])
        elif library.iloc[x, -1] == "None" and library.iloc[x, -2] == "None":
            col0.append(library.iloc[x, -2])
            col5.append(library.iloc[x, 14])
            col6.append(library.iloc[x, 15])
            col4.append(library.iloc[x, 16])
            col3.append(library.iloc[x, 17])
            col2.append(library.iloc[x, 18])
            col1.append(library.iloc[x, 19])
            colNR.append(library.iloc[x, 20])
            colN.append(library.iloc[x, 30])
            colO.append(library.iloc[x, 31])

    library_charges = pd.DataFrame({
        'C1': col1,
        'C2': col2,
        'C3': col3,
        'C4': col4,
        'C5': col5,
        'C6': col6,
        'C_NR': colNR,
        'N': colN,
        'O': colO
    }, index=col0)

    return library_charges

def description_sheet(of_what):
    """Function providing statistical description within rows
        and columns of a given dataframe

        Parameters
        ----------
        of_what : DataFrame
            name of a DataFrame to be described

        Returns
        -------
        description_rows: DataFrame
            Returns dataframe containing description for each column.
        description_columns: DataFrame
            Returns dataframe containing description for each column.
        """

    row_names = list(of_what.index.values)
    column_names = list(of_what.columns)

    mean_in_col = []
    deviation_in_col = []
    min_in_col = []
    max_in_col = []
    for x in range(0, len(of_what)):
        a, b, c, d, e, f, g, h = of_what[column_names[x]].describe()
        mean_in_col.append(b)
        deviation_in_col.append(c)
        min_in_col.append(d)
        max_in_col.append(h)

    mean_in_row = []
    deviation_in_row = []
    min_in_row = []
    max_in_row = []
    for x in range(0, len(of_what)):
        a, b, c, d, e, f, g, h = of_what.loc[row_names[x]].describe()
        mean_in_row.append(b)
        deviation_in_row.append(c)
        min_in_row.append(d)
        max_in_row.append(h)

    description_rows = pd.DataFrame({
        'Substituent': row_names,
        'Mean activation energy [kcal/mol]': mean_in_row,
        'Standard deviation': deviation_in_row,
        'Minimal value': min_in_row,
        'Maximal Value': max_in_row,
        'header_title': 'Distal effect'
    })

    description_columns = pd.DataFrame({
        'Substituent': column_names,
        'Mean activation energy [kcal/mol]': mean_in_col,
        'Standard deviation': deviation_in_col,
        'Minimal value': min_in_col,
        'Maximal Value': max_in_col,
        'header_title': 'Proximal effect'
    })
    description_columns.sort_index(ascending=False, inplace=True)

    return description_rows, description_columns


# !!!!!!!!!!!!!!!!Main file!!!!!!!!!!!!!!!!!!!!!!!
os.mkdir(r'./ChargeDist')
library = pd.read_csv("ScanLibrary.csv", index_col=0, header=0)
library.sort_index(inplace=True)
# Simplifying the None labels
for x in range(0, len(library)):
    if library.iloc[x, -1] == "None,None":
        library.iloc[x, -1] = "None"
    if library.iloc[x, -2] == "None,None":
        library.iloc[x, -2] = "None"

library_act_ene = sub_library_compiler('A ring substitution ID',
                                       'B ring substitution ID',
                                       'Activation energy - bridge A',
                                       'Activation energy - bridge B')
library_sec_min = sub_library_compiler('A ring substitution ID',
                                       'B ring substitution ID',
                                       'Second minimum - bridge A',
                                       'Second minimum - bridge B')
library_OH_length = sub_library_compiler('A ring substitution ID',
                                         'B ring substitution ID',
                                         'Ring A starting lenght',
                                         'Ring B starting lenght')
library_N_charges = sub_library_compiler('A ring substitution ID',
                                         'B ring substitution ID',
                                         'Bridge A nitrogen charge',
                                         'Bridge B nitrogen charge')
library_O_charges = sub_library_compiler('A ring substitution ID',
                                         'B ring substitution ID',
                                         'Bridge A oxygen charge',
                                         'Bridge B oxygen charge')
library_C5_charges = sub_library_compiler('A ring substitution ID',
                                          'B ring substitution ID',
                                          'A ring carbon 5 Hirschfeld charge',
                                          'B ring carbon 5 Hirschfeld charge')
library_conectorC_charges = sub_library_compiler('A ring substitution ID',
                                                 'B ring substitution ID',
                                                 'Connector carbon A Mulliken charge',
                                                 'Connector carbon B Mulliken charge')
library_NO_dist = sub_library_compiler('A ring substitution ID',
                                       'B ring substitution ID',
                                       'Bridge A nitrogen - oxygen distance',
                                       'Bridge B nitrogen - oxygen distance')
library_NC_dist = sub_library_compiler('A ring substitution ID',
                                       'B ring substitution ID',
                                       'Bridge A nitrogen - carbon 1 distance',
                                       'Bridge B nitrogen - carbon 1 distance')
library_CO_dist = sub_library_compiler('A ring substitution ID',
                                       'B ring substitution ID',
                                       'Bridge A carbon 4  - oxygen distance',
                                       'Bridge B carbon 4  - oxygen distance')
sub_position = sub_library_compiler('A ring substitution ID', 'B ring substitution ID',
                                    'Ring A substituted in postion',
                                    'Ring B substituted in postion')
sub_position = sub_position.fillna(0)
sub_group = sub_library_compiler('A ring substitution ID', 'B ring substitution ID',
                                 'Ring A substituent',
                                 'Ring B substituent')

# Preparing charge tables for proximal substitution

charge_H_table = sub_library_compiler_charges()

# Drawing images of charges distribution in the proximal part for each sub. pattern
for x in range(0, len(charge_H_table)):

    # Importing base image
    graph_base = Image.open("charges.png")

    # Setting fonts
    text_font = ImageFont.truetype('Arial/arialbd.ttf', 16)
    title_font = ImageFont.truetype('Arial/ariblk.ttf', 25)

    # Importing entries from table
    text_title = charge_H_table.index[x]
    text_C1 = 'C1:\n' + "{:.3f}".format(charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0])
    text_C2 = 'C2:\n' + "{:.3f}".format(charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1])
    text_C3 = 'C3:\n' + "{:.3f}".format(charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2])
    text_C4 = 'C4:\n' + "{:.3f}".format(charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3])
    text_C5 = 'C5:\n' + "{:.3f}".format(charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4])
    text_C6 = 'C6:\n' + "{:.3f}".format(charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5])
    text_CNR = 'C:\n' + "{:.3f}".format(charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6])
    text_N = 'N:\n' + "{:.3f}".format(charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7])
    text_O = 'O:\n' + "{:.3f}".format(charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8])

    # Setting editable
    graph_base_editable = ImageDraw.Draw(graph_base)

    # Placing text on image
    graph_base_editable.text((10, 10), text_title, (0, 0, 0), font=title_font)

    w, h = text_font.getsize(text_C1)
    graph_base_editable.rectangle(
        (120 - 4, 60 - 2, 120 + 50, 60 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0]) * 4000)))
    graph_base_editable.multiline_text((120, 60), text_C1, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_C2)
    graph_base_editable.rectangle(
        (60 - 4, 96 - 2, 60 + 50, 96 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1]) * 4000)))
    graph_base_editable.multiline_text((60, 96), text_C2, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_C3)
    graph_base_editable.rectangle(
        (60 - 4, 167 - 2, 60 + 50, 167 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2]) * 4000)))
    graph_base_editable.multiline_text((60, 167), text_C3, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_C4)
    graph_base_editable.rectangle(
        (120 - 4, 188 - 2, 120 + 50, 188 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3]) * 4000)))
    graph_base_editable.multiline_text((120, 188), text_C4, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_C5)
    graph_base_editable.rectangle(
        (180 - 4, 167 - 2, 180 + 50, 167 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4]) * 4000)))
    graph_base_editable.multiline_text((180, 167), text_C5, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_C6)
    graph_base_editable.rectangle(
        (180 - 4, 96 - 2, 180 + 50, 96 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5]) * 4000)))
    graph_base_editable.multiline_text((180, 96), text_C6, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_CNR)
    graph_base_editable.rectangle(
        (220 - 4, 55 - 2, 220 + 50, 55 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6]) * 4000)))
    graph_base_editable.multiline_text((220, 55), text_CNR, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_N)
    graph_base_editable.rectangle(
        (310 - 4, 133 - 2, 310 + 50, 133 + 2 * h + 4), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7]) * 4000)))
    graph_base_editable.multiline_text((310, 133), text_N, (0, 0, 0), align='center', font=text_font)

    w, h = text_font.getsize(text_O)
    graph_base_editable.rectangle(
        (270 - 4, 225 - 2, 270 + 50, 225 + 2 * h + 4), outline='Silver', fill=(
        150 + round((charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8]) * 4000)))
    graph_base_editable.multiline_text((270, 225), text_O, (0, 0, 0), align='center', font=text_font)

    # Saving result
    graph_base.save(r'./ChargeDist/' + charge_H_table.index[x] + 'charges.png')

for x in range(0, len(charge_H_table)):

    # Importing base image
    graph_base2 = Image.open("charges3.png")
    graph_base2 = graph_base2.crop((0, 0, 346, 250))
    # Setting fonts
    text_font = ImageFont.truetype('Arial/arialbd.ttf', 14)
    text_font2 = ImageFont.truetype('Arial/arialbd.ttf', 12)
    title_font = ImageFont.truetype('Arial/ariblk.ttf', 25)

    # Importing entries from table
    text_title = charge_H_table.index[x][0: -2] + ' in position ' + charge_H_table.index[x][-1]
    text_C1 = 'C1: ' + "{:.3f}".format(charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0])
    text_C2 = 'C2: ' + "{:.3f}".format(charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1])
    text_C3 = 'C3: ' + "{:.3f}".format(charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2])
    text_C4 = 'C4: ' + "{:.3f}".format(charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3])
    text_C5 = 'C5: ' + "{:.3f}".format(charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4])
    text_C6 = 'C6: ' + "{:.3f}".format(charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5])
    text_CNR = 'C: ' + "{:.3f}".format(charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6])
    text_N = 'N: ' + "{:.3f}".format(charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7])
    text_O = 'O: ' + "{:.3f}".format(charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8])

    # Setting editable
    graph_base_editable2 = ImageDraw.Draw(graph_base2)

    # Placing text on image
    graph_base_editable2.text((10, 10), text_title, (0, 0, 0), font=title_font)

    '''w, h = text_font.getsize(text_C1)
    graph_base_editable2.rectangle((5, 60, 5 + w, 60 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0]) * 4000)))'''
    graph_base_editable2.text((5, 60), text_C1, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_C2)
    graph_base_editable2.rectangle((5, 80, 5 + w, 80 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1]) * 4000)))'''
    graph_base_editable2.text((5, 80), text_C2, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_C3)
    graph_base_editable2.rectangle((5, 100, 5 + w, 100 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2]) * 4000)))'''
    graph_base_editable2.text((5, 100), text_C3, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_C4)
    graph_base_editable2.rectangle((5, 120, 5 + w, 120 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3]) * 4000)))'''
    graph_base_editable2.text((5, 120), text_C4, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_C5)
    graph_base_editable2.rectangle((5, 140, 5 + w, 140 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4]) * 4000)))'''
    graph_base_editable2.text((5, 140), text_C5, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_C6)
    graph_base_editable2.rectangle((5, 160, 5 + w, 160 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5]) * 4000)))'''
    graph_base_editable2.text((5, 160), text_C6, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_CNR)
    graph_base_editable2.rectangle((5, 180, 5 + w, 180 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6]) * 4000)))'''
    graph_base_editable2.text((5, 180), text_CNR, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_N)
    graph_base_editable2.rectangle((5, 200, 5 + w, 200 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7]) * 4000)))'''
    graph_base_editable2.text((5, 200), text_N, (0, 0, 0), font=text_font2)

    '''w, h = text_font.getsize(text_O)
    graph_base_editable2.rectangle((5, 220, 5 + w, 220 + h), outline='Black', fill=(
        150 + round((charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8]) * 4000),
        150,
        150 - round((charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8]) * 4000)))'''
    graph_base_editable2.text((5, 220), text_O, (0, 0, 0), font=text_font2)

    graph_base_editable2.regular_polygon(
        (148, 83, 15 + (charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 0] - charge_H_table.iloc[0, 0]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((143, 76), '1', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (98, 111, 15 + (charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 1] - charge_H_table.iloc[0, 1]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((93, 104), '2', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (98, 166, 15 + (charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 2] - charge_H_table.iloc[0, 2]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((93, 159), '3', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (148, 193, 15 + (charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 3] - charge_H_table.iloc[0, 3]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((143, 187), '4', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (198, 166, 15 + (charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 4] - charge_H_table.iloc[0, 4]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((193, 159), '5', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (198, 111, 15 + (charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 5] - charge_H_table.iloc[0, 5]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((193, 104), '6', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (246, 83, 15 + (charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 6] - charge_H_table.iloc[0, 6]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((241, 76), 'C', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (294, 109, 15 + (charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 7] - charge_H_table.iloc[0, 7]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((289, 102), 'N', (0, 0, 0), align='center', font=text_font)

    graph_base_editable2.regular_polygon(
        (240, 190, 15 + (charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8]) * 120), fill=(
            150 + round((charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8]) * 4000),
            150,
            150 - round((charge_H_table.iloc[x, 8] - charge_H_table.iloc[0, 8]) * 4000)),
        n_sides=300)
    graph_base_editable2.text((235, 183), 'O', (0, 0, 0), align='center', font=text_font)
    # Saving result
    graph_base2.save(r'./ChargeDist/scheme' + charge_H_table.index[x] + 'charges.png')

# Changing substitution position to integers
for x in range(0, len(sub_position)):
    for z in range(0, len(sub_position)):
        if sub_position.iloc[x, z] == "None":
            sub_position.iloc[x, z] = 0
for x in range(0, len(sub_position)):
    sub_position.iloc[:, x] = sub_position.iloc[:, x].astype('int')

activation_energy_unsub = library_act_ene.loc['None', 'None']

# Difference between Activation energy and second minimum of proton transfer
differences_of_ene = library_conectorC_charges.copy()  # Placeholder
for x in range(0, len(library_act_ene)):
    for z in range(0, len(library_act_ene)):
        differences_of_ene.iloc[x, z] = library_act_ene.iloc[x, z] - library_sec_min.iloc[x, z]

'''Describing effects of distal and proximal substitution:
distal_effect checks mean, standard deviation and extreme values for
each of the substitution schemes on proximal ring within values for a single substitution scheme on distal ring,
while proximal_effect - each of the substitution schemes on distal ring and a single one from proximal.
Then, the dataframe for each of the effects is built.
For example:
if we take NO2 in position 3 on proximal ring, we describe effects of different
substitution schemes form distal ring on it. If standard deviation is high, that means the effect
on the bridge from the distal ring substitution is considerable.'''

statistical = []
distal_effect_act_ene, proximal_effect_act_ene = description_sheet(library_act_ene)
statistical.append(distal_effect_act_ene)
statistical.append(proximal_effect_act_ene)
pd.concat(statistical).to_csv("Statistical_Activation.csv", index=False)

statistical = []
distal_effect_sec_min, proximal_effect_sec_min = description_sheet(library_sec_min)
statistical.append(distal_effect_sec_min)
statistical.append(proximal_effect_sec_min)
pd.concat(statistical).to_csv("Statistical_SecMin.csv", index=False)

statistical = []
distal_effect_N_charges, proximal_effect_N_charges = description_sheet(library_N_charges)
statistical.append(distal_effect_N_charges.rename(
    columns={'Mean activation energy [kcal/mol]': 'Charge on N'}))
statistical.append(proximal_effect_N_charges.rename(
    columns={'Mean activation energy [kcal/mol]': 'Charge on N'}))
pd.concat(statistical).to_csv("Statistical_N_charges.csv", index=False)

statistical = []
distal_effect_carbon5_charges, proximal_effect_carbon5_charges = description_sheet(library_C5_charges)
statistical.append(distal_effect_carbon5_charges.rename(
    columns={'Mean activation energy [kcal/mol]': 'Charge on C'}))
statistical.append(proximal_effect_carbon5_charges.rename(
    columns={'Mean activation energy [kcal/mol]': 'Charge on C'}))
pd.concat(statistical).to_csv("Statistical_C5_charges.csv", index=False)

# Drawing plots
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

sns.heatmap(library_act_ene, cmap='Reds', ax=axes[0, 0], vmin=5, vmax=10)
axes[0, 0].set_title('Activation energy [kcal/mol]')
axes[0, 0].set(xlabel='')

sns.heatmap(library_sec_min, cmap='Blues', ax=axes[0, 1], vmin=3, vmax=8)
axes[0, 1].set_title('Second minimum [kcal/mol]')
axes[0, 1].set(xlabel='', ylabel='')

sns.heatmap(differences_of_ene, cmap='Greens', ax=axes[0, 2])
axes[0, 2].set_title(r'$\Delta$(Act. E, 2nd Min.) [kcal/mol]')
axes[0, 2].set(xlabel='', ylabel='')

sns.heatmap(library_O_charges, cmap='Greys', ax=axes[1, 0])
axes[1, 0].set_title('Oxygen charge')
axes[1, 0].set(xlabel='')

sns.heatmap(library_N_charges, cmap='gray', ax=axes[1, 1])
axes[1, 1].set_title('Bridge nitrogen charge')
axes[1, 1].set(ylabel='')

sns.heatmap(library_OH_length, cmap='Greens', ax=axes[1, 2])
axes[1, 2].set_title(r'Starting O-H length [$\AA$]')
axes[1, 2].set(xlabel='', ylabel='')

plt.subplots_adjust(left=0.1,
                    bottom=0.125,
                    right=0.948,
                    top=0.94,
                    wspace=0.16,
                    hspace=0.28)
plt.savefig("figure1.png")

fig2, axes = plt.subplots(2, 3, figsize=(17, 10))
sns.heatmap(library_act_ene, cmap='Reds', ax=axes[0, 0], vmin=5, vmax=10)
axes[0, 0].set_title('Activation energy [kcal/mol]')
axes[0, 0].set(xlabel='')

sns.heatmap(library_sec_min, cmap='Blues', ax=axes[0, 1], vmin=3, vmax=8)
axes[0, 1].set_title('Second minimum [kcal/mol]')
axes[0, 1].set(xlabel='', ylabel='')

sns.heatmap(library_OH_length, cmap='Greens', ax=axes[0, 2])
axes[0, 2].set_title(r'Starting O-H length [$\AA$]')
axes[0, 2].set(xlabel='', ylabel='')

sns.heatmap(library_NO_dist, cmap='Greys', ax=axes[1, 0])
axes[1, 0].set_title(r'Starting O-N length [$\AA$]')
axes[1, 0].set(xlabel='')

sns.heatmap(library_NC_dist, cmap='Greys', ax=axes[1, 1])
axes[1, 1].set_title(r'Starting C1-N length [$\AA$]')
axes[1, 1].set(ylabel='')

sns.heatmap(library_CO_dist, cmap='Greys', ax=axes[1, 2])
axes[1, 2].set_title(r'Starting O-C4 length [$\AA$]')
axes[1, 2].set(xlabel='', ylabel='')

plt.subplots_adjust(left=0.1,
                    bottom=0.125,
                    right=0.948,
                    top=0.94,
                    wspace=0.16,
                    hspace=0.28)
plt.savefig("figure2.png")

fig3, axes = plt.subplots(2, 3, figsize=(17, 10))
sns.heatmap(library_act_ene, cmap='Reds', ax=axes[0, 0], vmin=5, vmax=10)
axes[0, 0].set_title('Activation energy [kcal/mol]')
axes[0, 0].set(xlabel='')

sns.heatmap(library_N_charges, cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Bridge nitrogen charge')
axes[0, 1].set(xlabel='', ylabel='')

sns.heatmap(library_O_charges, cmap='Greens', ax=axes[0, 2])
axes[0, 2].set_title('Oxygen charge')
axes[0, 2].set(xlabel='', ylabel='')

sns.heatmap(library_NO_dist, cmap='Reds', ax=axes[1, 0])
axes[1, 0].set_title(r'Starting O-N length [$\AA$]')
axes[1, 0].set(xlabel='')

sns.heatmap(library_NC_dist, cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title(r'Starting C1-N length [$\AA$]')
axes[1, 1].set(ylabel='')

sns.heatmap(library_CO_dist, cmap='Greens', ax=axes[1, 2])
axes[1, 2].set_title(r'Starting O-C4 length [$\AA$]')
axes[1, 2].set(xlabel='', ylabel='')

plt.subplots_adjust(left=0.1,
                    bottom=0.125,
                    right=0.948,
                    top=0.94,
                    wspace=0.16,
                    hspace=0.28)
plt.savefig("figure3.png")
print("Done")

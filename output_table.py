import os
import json
import argparse
import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, CustomJS, CheckboxButtonGroup, Button
from bokeh.models.widgets import DataTable, TextInput, TableColumn
from bokeh.io import output_file, show
from bokeh.events import ButtonClick

dl_code = """
const columns = Object.keys(source.data);
const nrows = source.data[columns[0]].length;
const lines = [columns.join(',')];

for (let i = 0; i < nrows; i++) {
    let row = [];
    for (let j = 0; j < columns.length; j++) {
        const column = columns[j];
        row.push(source.data[column][i].toString());
    }
    lines.push(row.join(','));
}

const filetext = lines.join("\\n");
const filename = 'mdap_brain_coral.csv';
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}
"""


overlay_div = Div(text="""
<link rel='stylesheet' type='text/css' href='output/style.css'>
                  
<a id="alink" href="" target="_blank">
<div class="container" id="c1">
    <img border="0" id="image" src="" width="1000px" height="auto">
    <div class="overlay"><img id="overlay_mask" src="" width="100%" height="100%"></div>
</div>
</a>
""")


# FIX this code to work with the new table, bk-data-table
table_select_image_code = """
var grid = document.getElementsByClassName('bk-data-table')[0];

if (grid==undefined){
    console.log('grid not found');
    
    active_row_ID = 0;
}
else
{
    var active_row = grid.querySelectorAll('.active')[0];

    if (active_row!=undefined){

        var active_row_ID = Number(active_row.children[0].innerText);

        for (var i=1, imax=active_row.children.length; i<imax; i++){
            if (active_row.children[i].className.includes('active')){
                var active_col_ID = i-1;
            }
        }

        console.log('row',active_row_ID);
        console.log('col',active_col_ID);
    }
}

document.getElementById("image").src = String(source.data['image'][active_row_ID]);
document.getElementById("image").style.opacity = 1;
document.getElementById("alink").href = "https://www.uahirise.org/"+String(source.data['name'][active_row_ID]);
document.getElementById("overlay_mask").src = String(source.data['mask'][active_row_ID]);

text_name.value = String(source.data['name'][active_row_ID]);
"""

def parse_args():
    parser = argparse.ArgumentParser()

    help_ = "Choose a directory to save data"
    parser.add_argument("-o", "--output", help=help_, type=str, default="output")
    parser.add_argument("-p", "--percent_tol", help="don't show results smaller than this % Area", type=float, default=5)

    return parser.parse_args()

if __name__ == "__main__":

    # parse command line arguments
    args = parse_args()

    # load statevector json from output directory
    with open(os.path.join(args.output, "statevector.json"), "rb") as f:
        sv = json.load(f)

# {'url': 'https://hirise-pds.lpl.arizona.edu/download/PDS/RDR/ESP/ORB_044500_044599/ESP_044505_2265/ESP_044505_2265_RED.JP2',
#  'mask_sum': 205,
#  'lat': 45.9184591456235,
#  'lon': 41.7063466340995,
#  'scale': 0.25,
#  'title': 'Layers in crater deposit in Protonilus Mensae',
#  'local_time': 14.91269,
#  'acq_date': '2016-01-24T08:16:05.301',
#  'segmentation_mask_sum': 9873215,
#  'segmentation_percent': 0.543835481386582}

    # load csv file 
    comments = pd.read_csv("Brain Coral Assessment - Revised_Table.csv")
    # 'name', 'title', 'comment', 'commenter', 'lat', 'lon', 'percent', 'resolution'

    # convert sv into dict of lists
    sv_dict = {
        'name':[],
        'title':[],
        'lon':[],
        'lat':[],
        'percent':[],
        'sum':[],
        'resolution':[],
        'comment':[],
        'commenter':[],

        # keys for image viewer
        'image':[],
        'mask':[],
    }

    for i, name in enumerate(sv['data'].keys()):

        if sv['data'][name].get('segmentation_percent',0) < args.percent_tol:
            continue

        sv_dict['name'].append(name.replace("_RED",""))
        sv_dict['title'].append(sv['data'][name]['title'])
        sv_dict['lon'].append(float(f"{sv['data'][name]['lon']:.5f}"))
        sv_dict['lat'].append(float(f"{sv['data'][name]['lat']:.5f}"))
        sv_dict['percent'].append(float(f"{sv['data'][name].get('segmentation_percent',0):.3f}"))
        sv_dict['sum'].append(sv['data'][name].get('segmentation_mask_sum',0))
        sv_dict['resolution'].append(sv['data'][name]['scale'])
       
        # check if comment exists
        if sv_dict['name'][-1] in comments['name'].values:
            idx = comments['name'].values.tolist().index(sv_dict['name'][-1])
            sv_dict['comment'].append(comments['comment'].values[idx])
            sv_dict['commenter'].append(comments['commenter'].values[idx])
        else:
            sv_dict['comment'].append("")
            sv_dict['commenter'].append("")

        # images
        sv_dict['image'].append(sv['data'][name]['url'].replace("download/","").replace("PDS","PDS/EXTRAS").replace("JP2","browse.jpg"))
        sv_dict['mask'].append(f"./{args.output}/{name}/{name}_classifier_mask.png")

    table_source = ColumnDataSource(sv_dict)

    columns = [
        TableColumn(field="name", title="Image"),
        TableColumn(field="title", title="Description"),
        TableColumn(field="lon", title="Longitude"),
        TableColumn(field="lat", title="Latitude"),
        TableColumn(field="percent", title="Coverage [%]"),
        TableColumn(field="resolution", title="Resolution [m/px]"),
        TableColumn(field="comment", title="Comment"),
        TableColumn(field="commenter", title="Commenter"),
    ]

    data_table = DataTable(source=table_source, columns=columns, width=1200, editable = True)

    # download table as csv button
    dl_button = Button(label="Download Table", button_type="success", width=500)
    dl_button.js_on_click(CustomJS(args=dict(source=table_source),code = dl_code))

    # image viewer below
    text_name = TextInput(value = "", title = "Name", width=500)

    # change image on table click
    callback = CustomJS(
        args = dict(source = table_source, text_name = text_name), 
        code = table_select_image_code)

    table_source.selected.js_on_change('indices', callback)

    # toggle buttons
    LABELS = ["Mask", "Zoom"]

    toggle_buttons = CheckboxButtonGroup(labels=LABELS, active=[])

    toggle_buttons.js_on_event(ButtonClick, CustomJS(args=dict(), code="""

        console.log('toggle_buttons: active=' + this.active, this.toString(), typeof(this.active))
        document.getElementById("overlay_mask").style.opacity = 0;
 
        document.getElementById("image").style.width = "1000px";
        document.getElementById("image").style.opacity = 1;

        if (this.active.includes(0)){
            document.getElementById("overlay_mask").style.opacity = 1;
        }
        if (this.active.includes(1)){
            document.getElementById("image").style.width = "auto";
        }
    """))

    output_file("index.html", title="HiRISE Brain Coral Table")

    show( 
        column(
            column(dl_button, data_table, text_name),
            column(toggle_buttons, overlay_div)
        )
    )
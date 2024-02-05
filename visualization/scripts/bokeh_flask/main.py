import json
import numpy as np
from sentence_transformers import SentenceTransformer
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, CustomJS
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models.widgets import CheckboxGroup, Select
import pandas as pd
import umap
from bokeh.models import CDSView, BooleanFilter
import torch

# flask
from bokeh.embed import components
from flask import Flask, render_template
import random
from instruct_features import MetricLength

app = Flask(__name__)

def set_seed(random_state):
    deterministic = True
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

set_seed(2)

def normalize_scores(scores):
    min_val = min(scores)
    max_val = max(scores)
    range_vals = max_val - min_val
    normalized_scores = [ (1/score) for score in scores] 
    return normalized_scores

@app.route('/')
@app.route('/index')
def homepage():

    metric = MetricLength()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    json_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/data/dolly/data.json'
    prompts_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/data/dolly/prompts.json'
    indices_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/sampled/dolly/llm_search/coreset/1000/sampled_llm_search_1000.parquet.gzip'
    clusters_path = '/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/samples/dolly_diverse_kmeans_14_1000.npy'

    with open(prompts_path, 'r') as f:
        dolly_prompts = json.load(f)

    info_pd = pd.read_csv(json_path)

    dolly_format_prompts = []
    for i in range(len(info_pd)):
        fsent = f"\###Instruction: {info_pd['instruction'][i]}"
        fsent += f"\n\n###Input: {info_pd['input'][i]}"
        fsent += f"\n\n#Response:"
        dolly_format_prompts.append(fsent)
    
    instructions = info_pd['instruction'].tolist()
    outputs = info_pd['output'].tolist()
    inputs = info_pd['input'].tolist()

    # indices_file = "/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/samples/1208/diverse_kmeans_complex_1k.npy" #1k
    indices_file = "/corpora/InstructTune/ait_single/active-instruction-tuning/datasets/samples/1203/diverse_kmeans_complex_3k.npy" #3k

    indices_df = pd.read_parquet(indices_path)

    # Perform an inner join to find matching rows based on the 'instruction' column
    matched_df = pd.merge(info_pd, indices_df, on=['instruction', 'output'], how='inner')

    # Extract the indices of the matching rows
    matched_indices = matched_df.index.tolist()
    coreset_indices = matched_indices

    indices_loaded = np.load(indices_file)
    indices = indices_loaded.flatten()

    sampled_indices = random.sample(list(indices), 100)
    sampled_coreset_indices = random.sample(list(coreset_indices), 100)

    selected_sampled = np.zeros(len(dolly_prompts), dtype=bool)
    selected_sampled[sampled_indices] = True

    selected_coreset = np.zeros(len(dolly_prompts), dtype=bool)
    selected_coreset[sampled_coreset_indices] = True

    non_selected = ~(selected_sampled | selected_coreset)

    # Convert the NumPy boolean arrays to lists for BooleanFilter
    selected_sampled_list = selected_sampled.tolist()
    selected_coreset_list = selected_coreset.tolist()
    non_selected_list = non_selected.tolist()


    reducer = umap.UMAP(n_neighbors=13, min_dist=1, random_state=2)

    dolly_embeddings = model.encode(dolly_format_prompts)
    umap_dolly_embeddings = reducer.fit_transform(dolly_embeddings)

    #Selected clusters
    clusters_np = np.load(clusters_path)
    selected_cluster_1 = (clusters_np[1]).tolist()
    selected_cluster_1_array = np.zeros(len(dolly_prompts), dtype=bool)
    selected_cluster_1_array[selected_cluster_1] = True
    selected_cluster_1_list = selected_cluster_1_array.tolist()

    final_1 = indices[3:6]
    print(final_1)
    final_1_array = np.zeros(len(dolly_prompts), dtype=bool)
    final_1_array[final_1] = True
    final_1_list = final_1_array.tolist()

    # Prepare data for bokeh
    info_source = ColumnDataSource(data=dict(
        x=umap_dolly_embeddings[:,0],
        y=umap_dolly_embeddings[:,1],
        input=inputs,
        output=outputs,
        instruction=instructions
    ))

    

    # print("Selected list sample:", selected_list[:10])
    # print("")
    # print("Selected list sample:", selected_list[10:20])
    # print("")

    # Create views for the selected and non-selected points
    view_sampled_selected = CDSView(source=info_source, filters=[BooleanFilter(selected_sampled_list)])
    view_coreset_selected = CDSView(source=info_source, filters=[BooleanFilter(selected_coreset_list)])
    view_non_selected = CDSView(source=info_source, filters=[BooleanFilter(non_selected_list)])
    view_selected_cluster_1 = CDSView(source=info_source, filters=[BooleanFilter(selected_cluster_1_list)])
    view_final_1 = CDSView(source=info_source, filters=[BooleanFilter(final_1_list)])

    # Configure hover tools
    TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,save"
    
    # Create figures
    # p1 = figure(width=850, height=800, tools=TOOLS,\
    #     title="Info Feature", x_axis_label='X1', y_axis_label='X2', toolbar_location="left")

    # Create glyphs
    # glyph_p1 = p1.circle('x', 'y', size=10, source=info_source, 
    #                  fill_color='blue', fill_alpha=0.6, 
    #                  line_color='black', legend_label='info')

    p1 = figure(width=1000, height=900, tools=TOOLS, title="Info Feature",
            x_axis_label='X1', y_axis_label='X2', toolbar_location="above")

    # Optionally, adjust the axis range to zoom out
    p1.x_range.start = min(umap_dolly_embeddings[:,0]) - 1  # Adjust as needed
    p1.x_range.end = max(umap_dolly_embeddings[:,0]) + 1    # Adjust as needed
    p1.y_range.start = min(umap_dolly_embeddings[:,1]) - 1  # Adjust as needed
    p1.y_range.end = max(umap_dolly_embeddings[:,1]) + 1  

    # Create glyphs
    # For non-selected points
    # glyph_p1 = p1.circle('x', 'y', size=10, source=info_source, color='white', 
    #                     fill_alpha=0.1, line_color='black', legend_label='info',
    #                     view=view_non_selected)

    # # For selected points
    # glyph_p1_selected = p1.circle('x', 'y', size=10, source=info_source, color='red', 
    #                             fill_alpha=0.4, line_color='black', legend_label='selected',
    #                             view=view_selected)
    
    glyph_non_selected = p1.cross('x', 'y', size=10, source=info_source, color='white', 
                              line_alpha=0.1, line_color='black', legend_label='Non-selected',
                              view=view_non_selected)

    # Create glyphs for sampled selected points
    glyph_sampled_selected = p1.cross('x', 'y', size=10, source=info_source, color='red', 
                                    line_alpha=0.8, line_color='red', legend_label='Sampled Selected',
                                    view=view_sampled_selected)

    # Create glyphs for coreset selected points
    glyph_coreset_selected = p1.cross('x', 'y', size=10, source=info_source, color='blue', 
                                    line_alpha=0.8, line_color='blue', legend_label='Coreset Selected',
                                    view=view_coreset_selected)
    
    glyph_selected_cluster_1 = p1.cross('x', 'y', size=10, source=info_source, color='red', 
                                    line_alpha=0.8, line_color='red', legend_label='Selected Cluster 1',
                                    view=view_selected_cluster_1)

    glyph_final_1 = p1.cross('x', 'y', size=12, source=info_source, color='green', 
                         line_alpha=1.0, line_color='green', legend_label='Final 1',
                         view=view_final_1)

    hover_tool_p1 = HoverTool(tooltips="""
    <div style="max-width: 300px;">
        <div>
            <span style="font-size 8px;">Instruction:</span>
            <span style="font-size: 10px;color: #696; word-wrap: break-word;">@instruction</span>
        </div>
        <div>
            <span style="font-size: 10px;">Input:</span>
            <span style="font-size: 10px;color: #696; word-wrap: break-word;">@input</span>
        </div>
        <div>
            <span style="font-size 8px;">Output:</span>
            <span style="font-size: 10px;color: #696; word-wrap: break-word;">@output{safe}</span>
        </div>
        <div>
            <span style="font-size: 10px;">Info Feature:</span>
            <span style="font-size: 10px; color: #696;">@info_feature</span>
        </div>
    </div>
""", renderers=[glyph_non_selected, glyph_sampled_selected, glyph_coreset_selected, glyph_selected_cluster_1, glyph_final_1])

    p1.add_tools(hover_tool_p1)

    p1.add_layout(p1.legend[0], 'right')

    p1.legend.click_policy = "hide"

    select = Select(title="Graph:", value="Info Feature", options=["Info Feature"])

    # Define a CustomJS callback to switch between graphs
    callback = CustomJS(args=dict(p1=p1, select=select), code="""
        p1.visible = select.value == "Info Feature";
    """)

    # Attach the callback to the Select widget
    select.js_on_change('value', callback)

    layout = column(select, p1)
    script, div = components(layout)
    return render_template(
        template_name_or_list='charts.html',
        script=script,
        div=div,
    )

if __name__ == '__main__':
    app.run(port=5001, debug=True)
    # homepage()
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import panel as pn
import umap
import torch
from metric_length import MetricLength

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, CustomJS
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models.widgets import CheckboxGroup, Select

# flask
from bokeh.embed import components
from flask import Flask, render_template
import random

app = Flask(__name__)

def normalize_scores(scores):
    min_val = min(scores)
    max_val = max(scores)
    range_vals = max_val - min_val
    normalized_scores = [ (1/score) for score in scores] 
    return normalized_scores


@app.route('/')
@app.route('/index')
def homepage():
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    with open('../../datasets/alpaca/alpaca_prompts_1000.jsonl', 'r') as f:
        alpaca_prompts = [json.loads(line)['prompt'] for line in f]

    with open('../../datasets/alpaca/alpaca_data_1000.jsonl', 'r') as f:
        alpaca_data = [json.loads(line) for line in f]

    with open('../../datasets/alpaca/alpaca_cleaned_new.jsonl', 'r') as f:
        alpaca_predicted = [line.strip() for line in f]

    with open('../scores/alpaca_cossim_scores.json', 'r') as f:
        alpaca_cossims = json.load(f)
    
    with open('../scores/alpaca_perplex_scores.json', 'r') as f:
        alpaca_perp_scores = json.load(f)
    
    with open('../scores/alpaca_rouge_scores.json', 'r') as f:
        alpaca_rouge_scores = json.load(f)

    with open('../scores/alpaca_rougel_scores.json', 'r') as f:
        alpaca_rougel_scores = json.load(f)

    with open('../../datasets/alpaca/alpaca_prompt_op_cossims.jsonl') as f:
        alpaca_prompt_op_cossims = json.load(f)

    # Load Lamini data
    with open('../../datasets/lamini/lamini_prompt_1000.jsonl', 'r') as f:
        lamini_prompts = [json.loads(line)['prompt'] for line in f]

    with open('../../datasets/lamini/lamini_1000.jsonl', 'r') as f:
        lamini_data = [json.loads(line) for line in f]

    with open('../../datasets/lamini/lamini_cleaned_new.jsonl', 'r') as f:
        lamini_predicted = [line.strip() for line in f]

    with open('../scores/lamini_cossim_scores.json', 'r') as f:
        lamini_cossims = json.load(f)    
    
    with open('../scores/lamini_perplex_scores.json', 'r') as f:
        lamini_perp_scores = json.load(f)
    
    with open('../scores/lamini_rouge_scores.json', 'r') as f:
        lamini_rouge_scores = json.load(f)

    with open('../scores/lamini_rougel_scores.json', 'r') as f:
        lamini_rougel_scores = json.load(f)

    with open('../../datasets/lamini/lamini_prompt_op_cossims.jsonl') as f:
        lamini_prompt_op_cossims = json.load(f)

    metafeature = MetricLength()
    
    alpaca_metric_length = metafeature.get_alpaca_length()
    lamini_metric_length = metafeature.get_lamini_length()

    lamini_perp_scores_norm = normalize_scores(lamini_perp_scores)
    alpaca_perp_scores_norm = normalize_scores(alpaca_perp_scores)

    reducer = umap.UMAP(random_state=2)

    alpaca_embeddings = model.encode(alpaca_prompts)
    umap_alpaca_embeddings = reducer.fit_transform(alpaca_embeddings)

    lamini_embeddings = model.encode(lamini_prompts)
    umap_lamini_embeddings = reducer.fit_transform(lamini_embeddings)

    alpaca_meta_vector = metafeature.meta_vector(alpaca_perp_scores_norm, alpaca_metric_length, alpaca_rougel_scores)
    lamini_meta_vector = metafeature.meta_vector(lamini_perp_scores_norm, lamini_metric_length, lamini_rougel_scores)

    #print(alpaca_meta_vector.shape)
    if torch.isnan(alpaca_meta_vector).any():
        # Handle NaN values (e.g., replace them with zeros or the mean of the column)
        alpaca_meta_vector = torch.nan_to_num(alpaca_meta_vector)
    if torch.isnan(lamini_meta_vector).any():
        # Handle NaN values (e.g., replace them with zeros or the mean of the column)
        lamini_meta_vector = torch.nan_to_num(lamini_meta_vector)


    alpaca_meta_vector = reducer.fit_transform(alpaca_meta_vector)
    lamini_meta_vector = reducer.fit_transform(lamini_meta_vector)

    #print(type(umap_lamini_embeddngs), type())
    # Prepare data for bokeh
    alpaca_source = ColumnDataSource(data=dict(
        x=umap_alpaca_embeddings[:,0],
        y=umap_alpaca_embeddings[:,1],
        input=[item['input'] for item in alpaca_data],
        output=[item['output'] for item in alpaca_data],
        instruction=[item['instruction'] for item in alpaca_data],
        predicted=alpaca_predicted,
        al_cossims=alpaca_cossims, alpaca_perp_scores = alpaca_perp_scores,
        alpaca_perp_scores_norm = alpaca_perp_scores_norm,
        alpaca_rougel_scores = alpaca_rougel_scores,
        alpaca_prompt_op_cossims = alpaca_prompt_op_cossims,
        alpaca_index= list(range(len(alpaca_prompt_op_cossims))),
        alpaca_metric_length = alpaca_metric_length,
        alpaca_meta_vector_x = alpaca_meta_vector[:, 0],
        alpaca_meta_vector_y = alpaca_meta_vector[:, 1]
    ))

    lamini_source = ColumnDataSource(data=dict(
        lx=umap_lamini_embeddings[:,0],
        ly=umap_lamini_embeddings[:,1],
        la_input=[item['instruction'] for item in lamini_data],
        la_output=[item['response'] for item in lamini_data],
        la_instruction_source=[item['instruction_source'] for item in lamini_data],
        la_predicted=lamini_predicted,
        la_cossims=lamini_cossims, lamini_perp_scores = lamini_perp_scores,
        lamini_perp_scores_norm = lamini_perp_scores_norm,
        lamini_rougel_scores = lamini_rougel_scores,
        lamini_prompt_op_cossims = lamini_prompt_op_cossims,
        lamini_index = list(range(len(lamini_prompt_op_cossims))),
        lamini_metric_length = lamini_metric_length,
        lamini_meta_vector_x = lamini_meta_vector[:, 0],
        lamini_meta_vector_y = lamini_meta_vector[:, 1]
    ))

    # Configure hover tools
    TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,save"
    # Create figure
    p1 = figure(width=850, height=800, tools=TOOLS,\
        title="Cossims", x_axis_label='X1', y_axis_label='X2', toolbar_location="left")

    p2 = figure(width=800, height=800, tools=TOOLS, \
    title="Perplexity", x_axis_label='X1', y_axis_label='X2', toolbar_location="left")

    p3 = figure(width=800, height=800, tools=TOOLS, \
    title="Rouge", x_axis_label='X1', y_axis_label='X2', toolbar_location="left")

    p4 = figure(width=800, height=800, tools=TOOLS, \
    title="Rouge vs 1/Perp.", x_axis_label='Rouge', y_axis_label='1/Perp.', toolbar_location="left")

    p5 = figure(width=800, height=800, tools=TOOLS, \
    title="Prompt vs Predictions", x_axis_label='Indices', y_axis_label='Cossims', toolbar_location="left")

    p6 = figure(width=800, height=800, tools=TOOLS, \
    title="Task Length", x_axis_label='Indices', y_axis_label='Length', toolbar_location="left")

    p7 = figure(width=800, height=800, tools=TOOLS, \
                title="Meta", x_axis_label='x1', y_axis_label='x2', toolbar_location="left")
    # Create glyphs
    alpaca_glyph_p1 = p1.circle('x', 'y', size=10, alpha='al_cossims', source=alpaca_source, color='blue', legend_label='alpaca')
    lamini_glyph_p1 = p1.circle('lx', 'ly', size=10, alpha='la_cossims', source=lamini_source, color='red', legend_label='lamini')

    alpaca_glyph_p2 = p2.circle('x', 'y', size=10, alpha='alpaca_perp_scores_norm', source=alpaca_source, color='blue', legend_label='alpaca')
    lamini_glyph_p2 = p2.circle('lx', 'ly', size=10, alpha='lamini_perp_scores_norm', source=lamini_source, color='red', legend_label='lamini')

    alpaca_glyph_p3 = p3.circle('x', 'y', size=10, alpha='alpaca_rougel_scores', source=alpaca_source, color='blue', legend_label='alpaca')
    lamini_glyph_p3 = p3.circle('lx', 'ly', size=10, alpha='lamini_rougel_scores', source=lamini_source, color='red', legend_label='lamini')

    alpaca_glyph_p4 = p4.circle('alpaca_rougel_scores', 'alpaca_perp_scores_norm', alpha='al_cossims', size=10, source=alpaca_source, color='blue', legend_label='alpaca')
    lamini_glyph_p4 = p4.circle('lamini_rougel_scores', 'lamini_perp_scores_norm', alpha='la_cossims', size=10, source=lamini_source, color='red', legend_label='lamini')

    alpaca_glyph_p5 = p5.circle('alpaca_index', 'alpaca_prompt_op_cossims', size=10, source=alpaca_source, color='blue', legend_label='alpaca')
    lamini_glyph_p5 = p5.circle('lamini_index', 'lamini_prompt_op_cossims', size=10, source=lamini_source, color='red', legend_label='lamini')

    alpaca_glyph_p6 = p6.circle('alpaca_index', 'alpaca_metric_length', size=10, alpha='al_cossims', source=alpaca_source, color='blue', legend_label='alpaca')
    lamini_glyph_p6 = p6.circle('lamini_index', 'lamini_metric_length', size=10, alpha='la_cossims', source=lamini_source, color='red', legend_label='lamini')

    alpaca_glyph_p7 = p7.circle('alpaca_meta_vector_x', 'alpaca_meta_vector_y', size=10, alpha='al_cossims', source=alpaca_source, color='blue', legend_label='alpaca')
    lamini_glyph_p7 = p7.circle('lamini_meta_vector_x', 'lamini_meta_vector_y', size=10, alpha='la_cossims', source=lamini_source, color='red', legend_label='lamini')
    # Create HoverTool
    # alpaca_hover_tool_p1 = HoverTool(tooltips=[
    #     ("input", "@input"),
    #     ("output", "@output"),
    #     ("instruction", "@instruction"),
    #     ("predicted", "@predicted"),
    #     ("Cosine Similarity Score:", "@al_cossims"),
    #     ("Perplexity:", "@alpaca_perp_scores")
    # ], renderers=[alpaca_glyph_p1])

    # Create HoverTool
    alpaca_hover_tool_p1 = HoverTool(tooltips="""
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
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity Score:</span>
                <span style="font-size: 10px; color: #696;">@al_cossims</span>
            </div>
        </div>
    """, renderers=[alpaca_glyph_p1])



    # lamini_hover_tool_p1 = HoverTool(tooltips=[
    #     ("input", "@la_input"),
    #     ("output", "@la_output"),
    #     ("instruction source", "@la_instruction_source"),
    #     ("predicted", "@la_predicted"),
    #     ("Cosine Similarity Score:", "@la_cossims"), ("Perplexity:", "@lamini_perp_scores")
    # ], renderers=[lamini_glyph_p1])

    lamini_hover_tool_p1 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size 8px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_input</span>
            </div>
            <div>
                <span style="font-size 8px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@la_predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Instruction Source:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_instruction_source</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity Score:</span>
                <span style="font-size: 10px; color: #696;">@la_cossims</span>
            </div>
        </div>
    """, renderers=[lamini_glyph_p1])

    # alpaca_hover_tool_p2 = HoverTool(tooltips=[
    #     ("input", "@input"),
    #     ("output", "@output"),
    #     ("instruction", "@instruction"),
    #     ("predicted", "@predicted"),
    #     ("Cosine Similarity Score:", "@al_cossims"),
    #     ("Perplexity:", "@alpaca_perp_scores")
    # ], renderers=[alpaca_glyph_p2])

    alpaca_hover_tool_p2 = HoverTool(tooltips="""
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
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Perplexity:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_perp_scores</span>
            </div>
        </div>
    """, renderers=[alpaca_glyph_p2])

    # lamini_hover_tool_p2 = HoverTool(tooltips=[
    #     ("input", "@la_input"),
    #     ("output", "@la_output"),
    #     ("instruction source", "@la_instruction_source"),
    #     ("predicted", "@la_predicted"),
    #     ("Cosine Similarity Score:", "@la_cossims"), ("Perplexity:", "@lamini_perp_scores")
    # ], renderers=[lamini_glyph_p2])

    lamini_hover_tool_p2 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size 8px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_input</span>
            </div>
            <div>
                <span style="font-size 8px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@la_predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Instruction Source:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_instruction_source</span>
            </div>
            <div>
                <span style="font-size: 10px;">Perplexity:</span>
                <span style="font-size: 10px; color: #696;">@lamini_perp_scores</span>
            </div>
        </div>
    """, renderers=[lamini_glyph_p2])

    alpaca_hover_tool_p3 = HoverTool(tooltips="""
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
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Rouge-L F1 Score:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_rougel_scores</span>
            </div>
        </div>
    """, renderers=[alpaca_glyph_p3])

    lamini_hover_tool_p3 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size 8px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_input</span>
            </div>
            <div>
                <span style="font-size 8px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@la_predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Instruction Source:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_instruction_source</span>
            </div>
            <div>
                <span style="font-size: 10px;">Rouge-L F1 Score:</span>
                <span style="font-size: 10px; color: #696;">@lamini_rougel_scores</span>
            </div>
        </div>
    """, renderers=[lamini_glyph_p3])

    alpaca_hover_tool_p4 = HoverTool(tooltips="""
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
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Rouge-L F1 Score:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_rougel_scores</span>
            </div>
            <div>
                <span style="font-size: 10px;">Perplexity:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_perp_scores</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity Score:</span>
                <span style="font-size: 10px; color: #696;">@al_cossims</span>
            </div>
        </div>
    """, renderers=[alpaca_glyph_p4])

    lamini_hover_tool_p4 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size 8px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_input</span>
            </div>
            <div>
                <span style="font-size 8px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@la_predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Instruction Source:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_instruction_source</span>
            </div>
            <div>
                <span style="font-size: 10px;">Rouge-L F1 Score:</span>
                <span style="font-size: 10px; color: #696;">@lamini_rougel_scores</span>
            </div>
            <div>
                <span style="font-size: 10px;">Perplexity:</span>
                <span style="font-size: 10px; color: #696;">@lamini_perp_scores</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity Score:</span>
                <span style="font-size: 10px; color: #696;">@la_cossims</span>
            </div>
        </div>
    """, renderers=[lamini_glyph_p4])

    alpaca_hover_tool_p5 = HoverTool(tooltips="""
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
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_prompt_op_cossims</span>
            </div>
        </div>
    """, renderers=[alpaca_glyph_p5])

    lamini_hover_tool_p5 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size 8px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_input</span>
            </div>
            <div>
                <span style="font-size 8px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@la_predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Instruction Source:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_instruction_source</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity:</span>
                <span style="font-size: 10px; color: #696;">@lamini_prompt_op_cossims</span>
            </div>
        </div>
    """, renderers=[lamini_glyph_p5])

    alpaca_hover_tool_p6 = HoverTool(tooltips="""
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
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_prompt_op_cossims</span>
            </div>
            <div>
                <span style="font-size: 10px;">Task Length:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_prompt_op_cossims</span>
            </div>
        </div>
    """, renderers=[alpaca_glyph_p6])

    lamini_hover_tool_p6 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size 8px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_input</span>
            </div>
            <div>
                <span style="font-size 8px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@la_predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Instruction Source:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_instruction_source</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity:</span>
                <span style="font-size: 10px; color: #696;">@lamini_prompt_op_cossims</span>
            </div>
            <div>
                <span style="font-size: 10px;">Task Length:</span>
                <span style="font-size: 10px; color: #696;">@lamini_metric_length</span>
            </div>
        </div>
    """, renderers=[lamini_glyph_p6])

    alpaca_hover_tool_p7 = HoverTool(tooltips="""
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
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Cosine Similarity:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_prompt_op_cossims</span>
            </div>
            <div>
                <span style="font-size: 10px;">Task Length:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_prompt_op_cossims</span>
            </div>
        </div>i
    """, renderers=[alpaca_glyph_p7])

    alpaca_hover_tool_p7 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size: 10px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@instruction</span>
            </div>
            <div>
                <span style="font-size: 10px;">Input:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@input</span>
            </div>
            <div>
                <span style="font-size: 10px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">X1:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@alpaca_meta_vector_x</span>
            </div>
            <div>
                <span style="font-size: 10px;">X2:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@alpaca_meta_vector_y</span>
            </div>
            <div>
                <span style="font-size: 10px;">Rouge-L F1 Score:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_rougel_scores</span>
            </div>
            <div>
                <span style="font-size: 10px;">Perplexity:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_perp_scores</span>
            </div> 
            <div>
                <span style="font-size: 10px;">Task Length:</span>
                <span style="font-size: 10px; color: #696;">@alpaca_prompt_op_cossims</span>
            </div>
        </div>
    """, renderers=[alpaca_glyph_p7])

    lamini_hover_tool_p7 = HoverTool(tooltips="""
        <div style="max-width: 300px;">
            <div>
                <span style="font-size: 10px;">Instruction:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_input</span>
            </div>
            <div>
                <span style="font-size: 10px;">Output:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_output{safe}</span>
            </div>
            <div>
                <span style="font-size: 10px;">Predicted:</span>
                <span style="font-size: 10px; color: #696; font-weight: bold; word-wrap: break-word;">@la_predicted</span>
            </div>
            <div>
                <span style="font-size: 10px;">Instruction Source:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@la_instruction_source</span>
            </div>
            <div>
                <span style="font-size: 10px;">X1:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@lamini_meta_vector_x</span>
            </div>
            <div>
                <span style="font-size: 10px;">X2:</span>
                <span style="font-size: 10px;color: #696; word-wrap: break-word;">@lamini_meta_vector_y</span>
            </div>
           <div>
                <span style="font-size: 10px;">Rouge-L F1 Score:</span>
                <span style="font-size: 10px; color: #696;">@lamini_rougel_scores</span>
            </div>
            <div>
                <span style="font-size: 10px;">Perplexity:</span>
                <span style="font-size: 10px; color: #696;">@lamini_perp_scores</span>
            </div> 
            <div>
                <span style="font-size: 10px;">Task Length:</span>
                <span style="font-size: 10px; color: #696;">@lamini_metric_length</span>
            </div>
        </div>
    """, renderers=[lamini_glyph_p7])

    # Add HoverTool to plot
    p1.add_tools(alpaca_hover_tool_p1, lamini_hover_tool_p1)
    p2.add_tools(alpaca_hover_tool_p2, lamini_hover_tool_p2)
    p3.add_tools(alpaca_hover_tool_p3, lamini_hover_tool_p3)
    p4.add_tools(alpaca_hover_tool_p4, lamini_hover_tool_p4)
    p5.add_tools(alpaca_hover_tool_p5, lamini_hover_tool_p5)
    p6.add_tools(alpaca_hover_tool_p6, lamini_hover_tool_p6)
    p7.add_tools(alpaca_hover_tool_p7, lamini_hover_tool_p7)


    # p1.legend.location = "top_right"
    # p1.legend.orientation = "vertical"

    # p2.legend.location = "top_right"
    # p2.legend.orientation = "vertical"

    p1.add_layout(p1.legend[0], 'right')
    p2.add_layout(p2.legend[0], 'right')
    p3.add_layout(p3.legend[0], 'right')
    p4.add_layout(p4.legend[0], 'right')
    p5.add_layout(p5.legend[0], 'right')
    p6.add_layout(p6.legend[0], 'right')
    p7.add_layout(p7.legend[0], 'right')

    p1.legend.click_policy = "hide"
    p2.legend.click_policy = "hide"
    p3.legend.click_policy = "hide"
    p4.legend.click_policy = 'hide'
    p5.legend.click_policy = 'hide'
    p6.legend.click_policy = 'hide'
    p7.legend.click_policy = 'hide'

    p2.visible = False
    p3.visible = False
    p4.visible = False
    p5.visible = False
    p6.visible = False
    p7.visible = False

    select = Select(title="Graph:", value="Cossims", options=["Cossims", "Perplexity", "Rouge","Rouge vs 1/Perp.","Prompt vs Predictions", "Task Length", "Meta"])

    # Define a CustomJS callback to switch between graphs
    callback = CustomJS(args=dict(p1=p1, p2=p2, p3=p3, p4=p4, p5=p5,p6=p6,p7 = p7, select=select), code="""
        p1.visible = select.value == "Cossims";
        p2.visible = select.value == "Perplexity";
        p3.visible = select.value == "Rouge";
        p4.visible = select.value == "Rouge vs 1/Perp.";
        p5.visible = select.value == "Prompt vs Predictions";
        p6.visible = select.value == "Task Length";
        p7.visible = select.value == "Meta";
    """)

    # Attach the callback to the Select widget
    select.js_on_change('value', callback)

    # Create a Panel column with the plots
    plots = pn.Column(p1, p2, p3, p4, p5, p6, p7)

    # Create a Panel layout with the widget box and the plots
    layout = pn.Column(select, plots)

    # Convert the Panel layout to a Bokeh layout
    layout = layout.get_root()


    script, div = components(layout)
    return render_template(
        template_name_or_list='charts.html',
        script=script,
        div=div,
    )

if __name__ == '__main__':
    app.run(port=5001, debug=True)

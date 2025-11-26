"""
Enhanced Visualization Dashboard for PrxteinMPNN Benchmarks.
Features: Dark mode, Bootstrap layout, Violin plots, Pareto frontiers, and Heatmaps.
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

# --- Configuration & Theming ---
THEME = dbc.themes.CYBORG  # Dark, scientific theme
COLOR_PALETTE = px.colors.qualitative.Bold
MD_COLOR = "#FF595E"  # Reddish for Heat/MD
GAUSSIAN_COLOR = "#1982C4"  # Blueish for Noise

def load_data():
    """Load benchmark data with robust error handling."""
    geo_df = pd.read_csv("benchmark_geometric_integrity.csv") if os.path.exists("benchmark_geometric_integrity.csv") else None
    rec_df = pd.read_csv("benchmark_sequence_recovery.csv") if os.path.exists("benchmark_sequence_recovery.csv") else None
    
    # Pre-processing for nicer labels
    if rec_df is not None:
        # Create a unified label column if columns exist, otherwise default
        method = rec_df['noise_method'] if 'noise_method' in rec_df else "Unknown"
        param = rec_df['noise_param'].astype(str) if 'noise_param' in rec_df else "?"
        rec_df['method_label'] = method + " (" + param + ")"
    
    return geo_df, rec_df

geo_df, rec_df = load_data()

app = Dash(__name__, external_stylesheets=[THEME])

# --- Reusable Components ---

def make_card(title, content_id, controls=None, description=None):
    """Generate a consistent card container for graphs."""
    return dbc.Card([
        dbc.CardHeader(html.H5(title, className="mb-0")),
        dbc.CardBody([
            html.P(description, className="text-muted small") if description else None,
            controls if controls else html.Div(),
            dcc.Loading(dcc.Graph(id=content_id, style={"height": "450px"}))
        ])
    ], className="mb-4 shadow-sm")

# --- Layout ---

sidebar = html.Div([
    html.H2("PrxteinMPNN", className="display-6"),
    html.Hr(),
    html.P("Benchmark Analytics", className="lead"),
    dbc.Nav([
        dbc.NavLink("Geometric Integrity", href="#geo-section", active="exact"),
        dbc.NavLink("Sequence Recovery", href="#rec-section", active="exact"),
        dbc.NavLink("Pareto Frontier", href="#pareto-section", active="exact"),
    ], vertical=True, pills=True),
], style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "16rem", "padding": "2rem 1rem", "backgroundColor": "#111"})

content = html.Div([
    # SECTION 1: GEOMETRIC INTEGRITY
    html.H3("1. Geometric Integrity (Physics)", id="geo-section", className="mt-4 mb-3"),
    dbc.Row([
        dbc.Col([
            make_card(
                "Metric Distribution (Violin)", 
                "geo-plot",
                controls=dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id='geo-metric-dropdown',
                        options=[
                            {'label': 'Mean Bond Deviation (Å)', 'value': 'mean_bond_dev'},
                            {'label': 'Max Bond Deviation (Å)', 'value': 'max_bond_dev'},
                            {'label': 'Total Energy (kcal/mol)', 'value': 'total_energy'}
                        ],
                        value='mean_bond_dev',
                        clearable=False
                    ), width=6),
                    dbc.Col(dbc.Checklist(
                        id='geo-options',
                        options=[{'label': 'Log Scale', 'value': 'log'}],
                        value=[],
                        switch=True,
                        inline=True
                    ), width=6, className="mt-2")
                ]),
                description="Violin plots show the probability density of the data. Wider sections represent more samples."
            )
        ], width=12),
    ]),

    # SECTION 2: SEQUENCE RECOVERY
    html.H3("2. Sequence Recovery (Biology)", id="rec-section", className="mt-5 mb-3"),
    dbc.Row([
        dbc.Col([
            make_card(
                "Recovery Trends", 
                "recovery-trend-plot",
                controls=dcc.Dropdown(
                    id="rec-group-dropdown",
                    options=[
                        {"label": "Group by Method", "value": "noise_method"},
                        {"label": "Group by Model", "value": "model"}
                    ],
                    value="noise_method",
                    clearable=False
                ),
                description="Lines represent mean recovery; shaded areas indicate standard deviation."
            )
        ], width=8),
        dbc.Col([
            make_card(
                "Parameter Heatmap", 
                "recovery-heatmap",
                description="Optimal region: Intersection of Sampling Temp (X) and Noise Param (Y)."
            )
        ], width=4),
    ]),

    # SECTION 3: PARETO FRONTIER
    html.H3("3. The Trade-off (Pareto Frontier)", id="pareto-section", className="mt-5 mb-3"),
    dbc.Row([
        dbc.Col([
            make_card(
                "Recovery vs. Physical Validity", 
                "pareto-plot",
                description="Top-Left is ideal (High Recovery, Low Bond Deviation). Points sized by Perplexity.",
                controls=html.Div(id="pareto-controls") # Placeholder
            )
        ], width=12)
    ])

], style={"marginLeft": "18rem", "marginRight": "2rem", "padding": "2rem"})

app.layout = html.Div([sidebar, content])

# --- Callbacks ---

@app.callback(
    Output('geo-plot', 'figure'),
    [Input('geo-metric-dropdown', 'value'),
     Input('geo-options', 'value')]
)
def update_geo_plot(metric, options):
    if geo_df is None: return px.scatter(title="No Data")
    
    df = geo_df.copy()
    if metric == 'total_energy': 
        df = df[df['is_finite']]
    
    log_y = 'log' in options
    
    # Use Violin plots for better density visualization
    fig = go.Figure()
    
    methods = df['method'].unique()
    for m in methods:
        sub_df = df[df['method'] == m]
        fig.add_trace(go.Violin(
            x=sub_df['method'],
            y=sub_df[metric],
            name=m,
            box_visible=True,
            meanline_visible=True,
            line_color=MD_COLOR if 'MD' in m else GAUSSIAN_COLOR
        ))

    fig.update_layout(
        template="plotly_dark",
        yaxis_type="log" if log_y else "linear",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=1.1)
    )
    return fig

@app.callback(
    [Output("recovery-trend-plot", "figure"),
     Output("recovery-heatmap", "figure")],
    [Input("rec-group-dropdown", "value")]
)
def update_recovery_visuals(group_by):
    if rec_df is None: return px.scatter(title="No Data"), px.scatter(title="No Data")
    
    # 1. Trend Line with Error Bands (Better than Box plots for trends)
    # Aggregate data first
    agg_df = rec_df.groupby(['temperature', group_by])['recovery'].agg(['mean', 'std']).reset_index()
    
    fig_trend = px.line(
        agg_df, 
        x='temperature', 
        y='mean', 
        color=group_by, 
        markers=True,
        error_y='std',
        title="Recovery Trajectory"
    )
    fig_trend.update_layout(template="plotly_dark", hovermode="x unified")
    
    # 2. Heatmap (Inference Temp vs Noise Param)
    # We focus on the 'MD' method for the heatmap if mixed data exists, or just the first method
    heatmap_df = rec_df[rec_df['noise_method'] == 'MD'].copy() if 'MD' in rec_df['noise_method'].values else rec_df.copy()
    
    # Pivot for heatmap matrix
    # Averaging over other dimensions like 'model' or 'ensemble' to get a clean 2D surface
    pivot_table = heatmap_df.groupby(['noise_param', 'temperature'])['recovery'].mean().unstack()
    
    fig_heat = px.imshow(
        pivot_table,
        labels=dict(x="Sampling Temp", y="MD Temp / Noise", color="Recovery"),
        x=pivot_table.columns,
        y=pivot_table.index,
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    fig_heat.update_layout(template="plotly_dark", title="Parameter Sweet Spot")
    
    return fig_trend, fig_heat

@app.callback(
    Output("pareto-plot", "figure"),
    Input("geo-metric-dropdown", "value") # Dummy trigger, could add specific controls
)
def update_pareto(trigger):
    if geo_df is None or rec_df is None: return px.scatter(title="No Data")
    
    # We need to merge datasets. 
    # Assuming both CSVs have 'noise_param' and 'method' columns to link them roughly.
    # In reality, you might link by 'pdb' or run ID. 
    # Here we aggregate by method/param to show the "Global" performance of that setting.
    
    g_agg = geo_df.groupby(['method', 'param'])['mean_bond_dev'].mean().reset_index()
    r_agg = rec_df.groupby(['noise_method', 'noise_param'])['recovery'].mean().reset_index()
    
    # Normalize column names for merge
    g_agg.rename(columns={'param': 'p', 'method': 'm'}, inplace=True)
    r_agg.rename(columns={'noise_param': 'p', 'noise_method': 'm'}, inplace=True)
    
    # Ensure types match for merge
    g_agg['p'] = g_agg['p'].astype(str)
    r_agg['p'] = r_agg['p'].astype(str)
    
    merged = pd.merge(g_agg, r_agg, on=['m', 'p'])
    
    fig = px.scatter(
        merged,
        x="mean_bond_dev",
        y="recovery",
        color="m",
        size="recovery", # Just visual flair
        text="p",
        title="Pareto Frontier: Physics vs. Biology",
        labels={"mean_bond_dev": "Physical Violation (Lower is Better)", "recovery": "Sequence Recovery (Higher is Better)"}
    )
    
    # Invert X axis because lower deviation is better
    fig.update_xaxes(autorange="reversed")
    
    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_dark")
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)

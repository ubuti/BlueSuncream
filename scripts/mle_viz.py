# Pseudocode:
# 1. Import necessary libraries: numpy, dash, dash_core_components, dash_html_components, plotly.graph_objs
# 2. Generate sample data from a normal distribution
# 3. Define the normal PDF function
# 4. Define the log-likelihood function
# 5. Initialize the Dash app
# 6. Define the layout with sliders for mean (μ) and standard deviation (σ), display for log-likelihood, and a graph
# 7. Create a callback function that updates the log-likelihood and the plot based on slider inputs
# 8. In the callback, compute the log-likelihood and update the distribution curve and data points
# 9. Run the Dash app

# Python Code:
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ----------------------------- #
# 1. Data Generation
# ----------------------------- #

# Sample data: Normally distributed data points
np.random.seed(42)
true_mu = 0
true_sigma = 1
data = np.random.normal(loc=true_mu, scale=true_sigma, size=100)

# ----------------------------- #
# 2. Log-Likelihood Function
# ----------------------------- #

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(data, mu, sigma):
    return np.sum(np.log(normal_pdf(data, mu, sigma)))

# ----------------------------- #
# 3. Initialize Dash App
# ----------------------------- #

app = dash.Dash(__name__)
app.title = "Maximum Likelihood Estimation (MLE) Visualization"

# ----------------------------- #
# 4. Define App Layout
# ----------------------------- #

app.layout = html.Div([
    html.H1("Maximum Likelihood Estimation (MLE) Interactive Visualization"),
    
    html.Div([
        html.Div([
            html.Label("Mean (μ):"),
            dcc.Slider(
                id='mu-slider',
                min=-5,
                max=5,
                step=0.1,
                value=0,
                marks={i: str(i) for i in range(-5, 6)}
            ),
            html.Div(id='mu-value', style={'textAlign': 'center'})
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20'}),
        
        html.Div([
            html.Label("Standard Deviation (σ):"),
            dcc.Slider(
                id='sigma-slider',
                min=0.1,
                max=5,
                step=0.1,
                value=1,
                marks={i: str(i) for i in range(1, 6)}
            ),
            html.Div(id='sigma-value', style={'textAlign': 'center'})
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20'})
    ]),
    
    html.Div([
        html.H3("Log Likelihood:"),
        html.Div(id='log-likelihood', style={'textAlign': 'center'})
    ], style={'padding': '20px'}),
    
    dcc.Graph(id='mle-graph')
])

# ----------------------------- #
# 5. Define Callback
# ----------------------------- #

@app.callback(
    [Output('mu-value', 'children'),
     Output('sigma-value', 'children'),
     Output('log-likelihood', 'children'),
     Output('mle-graph', 'figure')],
    [Input('mu-slider', 'value'),
     Input('sigma-slider', 'value')]
)
def update_visualization(mu, sigma):
    # Update display values
    mu_text = f"μ = {mu}"
    sigma_text = f"σ = {sigma}"
    
    # Compute log-likelihood
    ll = log_likelihood(data, mu, sigma)
    ll_text = f"{ll:.2f}"
    
    # Generate distribution curve
    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    y_vals = normal_pdf(x_vals, mu, sigma)
    
    # Create Plotly traces
    trace_pdf = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='blue')
    )
    
    trace_data = go.Scatter(
        x=data,
        y=np.zeros_like(data),
        mode='markers',
        name='Data Points',
        marker=dict(color='red', size=8, opacity=0.6)
    )
    
    # Define layout
    layout = go.Layout(
        title="Normal Distribution and Log-Likelihood",
        xaxis=dict(title='x'),
        yaxis=dict(title='Probability Density'),
        hovermode='closest'
    )
    
    # Combine traces
    figure = {
        'data': [trace_pdf, trace_data],
        'layout': layout
    }
    
    return mu_text, sigma_text, ll_text, figure

# ----------------------------- #
# 6. Run the App
# ----------------------------- #

if __name__ == '__main__':
    app.run_server(debug=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dash
"""
import dash
import dash_html_components as html
import dash_core_components as dcc
import WHOLE_MODEL as WM # our prediction model
from WHOLE_MODEL import *
WP = WM.webpage()
app = dash.Dash()

style0={'width': '100%','height':'200'}
style1={'width': '100%','height':'300'}

direction = 'Press press "submit" first...'
sum_direc = 'Summary...'
exc_direc = 'Excerpt...'

app.layout = html.Div([
    html.H1(children='Book Genre Classification System'),
    
    html.H2(children='-- CSE 881 Project'),
    
    html.H3(children='by Lyuxun Yang, Yimin Wu, Qianqian Zhao'),
    
    html.Div(children='''
        Please paste the Summary and/or the Excerpt, and click "Submit".
    '''),
    
    dcc.Textarea(id='sum-input-box',
    placeholder=sum_direc,
    value=sum_direc,
    style=style0),

    dcc.Textarea(id='exc-input-box',
    placeholder=exc_direc,
    value=exc_direc,
    style=style0),
                 
    html.Button('Submit', id='button'),
    
    html.H3(children='Results:'),
    
    dcc.Textarea(id='output-box',readOnly=True,
    placeholder=direction,
    value=direction,
    style=style0)
], style={'columnCount': 2})

@app.callback(
    dash.dependencies.Output('output-box', 'value'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('sum-input-box', 'value'),
    dash.dependencies.State('exc-input-box', 'value')])
def update_output(n_clicks, v1,v2):
    fictxt,tagtxt,ntags = WP.predict_all(v1,v2)
    if n_clicks is None:
        return direction
    if v1==sum_direc and v2==exc_direc:
        return "You haven't changed anything!"
    txt = 'This is a ['+fictxt+'], '
    txt += 'it belongs to the following '+str(ntags)+' genres:\n\n'
    txt += tagtxt
    return txt


if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0',port=8000)
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import * as serviceWorker from './serviceWorker';
import {AppBar, Toolbar, Typography} from '@material-ui/core';

ReactDOM.render(
  <React.StrictMode>
    <AppBar position="static" className="appbar" style={{backgroundColor: '#272727', width:'100%'}}>
        <Toolbar>
            <img src="logo.png" className="logo"></img>
        </Toolbar>
    </AppBar>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();

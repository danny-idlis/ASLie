import React, { Component } from 'react';
import * as handpose from '@tensorflow-models/handpose';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import { version_wasm } from '@tensorflow/tfjs-backend-wasm';
import serverProxy from './serverProxy';
import styled from 'styled-components';
import { VideoContainer, AppContainer, Output } from './ui-components'
import './App.css'
import { complexWithEvenIndex } from '@tensorflow/tfjs-core/dist/backends/complex_util';

const tf = require('@tensorflow/tfjs');
const historyThreshold = 20;
const framesThreshold = 12;
const threshold = 0.3;

const Canvas = styled.canvas`
    transform: rotateY(180deg);
    -webkit-transform:rotateY(180deg);
    -moz-transform:rotateY(180deg); 
`

const CANVAS_HEIGHT = 670;
const CANVAS_WIDTH = 900;


class App extends Component {
  state = {
    text: '',
    predictionManagement: {
      history: [],
      letter: null,
      counter: null
    }
  }

  componentDidMount() {
    this.setState({
      video: document.querySelector("#videoElement"),
      canvas: document.getElementById("canvasElement"),
      ctx: null,
      proxy: new serverProxy(),
      text: ""
    })

    this.getCameraAccess().then(async (result) => {
      //this.initializePredictor()
      let canvas = document.getElementById("canvasElement");
      let ctx = canvas.getContext('2d');
      ctx.translate(canvas.width, 0);
      //Flip the context horizontally 
      ctx.scale(-1, 1);
      this.registerEvents()
    })
  }

  registerEvents = () => {
    document.addEventListener('keydown', (e) => {
      if (e.keyCode === 8) {
        this.backSpace()
      }

      if (e.keyCode === 32) {
        this.space();
      }

      e.preventDefault();
    })
  }

  space = () => {
    if (this.state.text.length !== 0) {
      this.setState(prevState => {
        return {
          text: prevState.text + " "
        }
      })
    }
  }

  backSpace = () => {
    if (this.state.text.length !== 0) {
      this.setState(prevState => {
        return {
          text: prevState.text.slice(0,-1)
        }
      })
    }
  }

  initializePredictor = async () => {
    await tf.setBackend("webgl")
    const model = await handpose.load();
    this.setState({ model });
  }

  predict = async () => {
    //Draw the frames obtained from video stream on a canvas   
    this.state.ctx.drawImage(this.state.video, 0, 0, this.state.canvas.width, this.state.canvas.height);

    //Predict landmarks in hand in the frame of a video 
    const predictions = await this.state.model.estimateHands(this.state.video);

    if (predictions.length > 0 && predictions[0].handInViewConfidence > 0.8) {
      const landmarks = predictions[0].landmarks;
      this.displayImagesAtFingerTop(landmarks, this.state.video);
      let classification = await this.state.proxy.getLetter(landmarks);
      if (classification && classification.data)
        this.addClassificationToState(classification.data);
    }
    requestAnimationFrame(this.predict);
  }

  addClassificationToState = ({ letter, score }) => {
    const predManager = this.state.predictionManagement;
    if (score > threshold) predManager.history.push(letter);
    if (predManager.history.length > historyThreshold) {
      predManager.history.shift();
    }
    var found = undefined;
    var counts = {};
    predManager.history.forEach(x => counts[x] = (counts[x] || 0) + 1);
    console.table(counts);
    if (predManager.history.length > framesThreshold) {
      Object.keys(counts).forEach(x => {
        if (counts[x] > framesThreshold) {
          console.log(`Found ${counts[x]}/${predManager.history.length} instances of ${x}.`);
          found = x;
        }
      });
    }

    if (found) {
      this.setState(prevState => {
        predManager.history = [];
        let predictionManagement = predManager;
        return {
          ...prevState,
          text: prevState.text + found,
          predictionManagement
        }
      });
    } else {
      this.setState(prevState => {
        let predictionManagement = predManager;
        return {
          ...prevState,
          predictionManagement
        }
      });
    }

    return;

    // if (this.state.predictionManagement.letter !== classification) {
    //   console.log(`starting new count with ${classification}`)
    //   this.setState({
    //     predictionManagement: {
    //       letter: classification,
    //       counter: 1
    //     }
    //   })
    //   return;
    // }

    // if (this.state.predictionManagement.letter === classification &&
    //   this.state.predictionManagement.counter < framesThreshold) {
    //   console.log(`countinuing with ${classification} threshold at ${this.state.predictionManagement.counter}/${framesThreshold}`)
    //   this.setState(prevState => {
    //     let predictionManagement = prevState.predictionManagement;
    //     predictionManagement.counter++;
    //     return {
    //       ...prevState,
    //       predictionManagement
    //     }
    //   })
    //   return;
    // }

    // if (this.state.predictionManagement.letter === classification &&
    //   this.state.predictionManagement.counter >= framesThreshold) {
    //   console.log(`${classification} reached the threshold, printing it`)
    //   this.setState(prevState => {
    //     let predictionManagement = prevState.predictionManagement;
    //     predictionManagement.counter = 0;
    //     predictionManagement.letter = null;
    //     return {
    //       ...prevState,
    //       text: prevState.text + classification,
    //       predictionManagement
    //     }
    //   })
    //   return;
    // }
  }
  displayImagesAtFingerTop = (landmarks, video) => {
    for (let i = 0; i < landmarks.length; i++) {
      const x = (landmarks[i][0] / video.videoWidth) * CANVAS_WIDTH;
      const y = (landmarks[i][1] / video.videoHeight) * CANVAS_HEIGHT;
      this.state.ctx.fillRect(x, y, 5, 5)
    }
  }

  getCameraAccess = () => {
    return new Promise((res, rej) => {
      let video = document.querySelector("#videoElement");
      let canvas = document.getElementById("canvasElement");
      let ctx;
      if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(function (stream) {
            video.srcObject = stream;
            res(true)
          })
          .catch(function (err0r) {
            console.log("Something went wrong!");
            rej(false)
          });
      }
      video.onloadedmetadata = async () => {
        await this.initializePredictor()
        //Get the 2D graphics context from the canvas element 
        ctx = canvas.getContext('2d');
        //Reset the point (0,0) to a given point 
        ctx.translate(canvas.width, 0);
        //Flip the context horizontally 
        ctx.scale(-1, 1);
        ctx.fillStyle = "#FF0000";
        this.setState({ ctx }, () => {
          requestAnimationFrame(this.predict);
        })
      };
    })
  }

  resetText = () => {
    this.setState({
      text: ''
    })
  }



  render() {
    return (
      <AppContainer className="App">
        <VideoContainer>
          <video style={{ display: "none" }} autoPlay={true} id="videoElement"></video>
          <Canvas id="canvasElement" width={CANVAS_WIDTH} height={CANVAS_HEIGHT} style={{ boxShadow: '0 0 6px black' }}></Canvas>
        </VideoContainer>
        <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
          <Output>
            <h2>{this.state.text}</h2>
          </Output>
          <button onClick={() => { this.resetText() }}>Reset</button>
        </div>
      </AppContainer>
    );
  }
}

export default App;

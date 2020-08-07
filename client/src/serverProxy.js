import axios from 'axios';

class serverProxy {
     serverURL = 'http://127.0.0.1:5000'

     getLetter = async (keyPoints)=> {
        // Default options are marked with *
        try{
            let response = await axios.post(this.serverURL,keyPoints.map(k => [k[0],k[1]]))
            return response
        }
        catch(error) {
            console.log(error)
        }
      }

    getClassification = async (clientPrediction) => {
        let letters = ["A", "B", "C", "D", "E", "F"];
        const getRandomInt =(min, max) => {
            min = Math.ceil(min);
            max = Math.floor(max);
            return Math.floor(Math.random() * (max - min)) + min; //The maximum is exclusive and the minimum is inclusive
          };
          let index = getRandomInt(0,5);
          return letters[index]
    }
}

export default serverProxy;
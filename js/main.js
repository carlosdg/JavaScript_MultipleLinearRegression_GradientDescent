/*
 * @author  Carlos Domínguez García
 * @date    19/07/2017
 * @brief   Implementation of linear regression with multiple independent variables
 *          using Gradient Descent to get an aproximation to the parameters
 *          of the linear function using the least square error as the cost
 *          function.
 * @note    In the vectors of independent variables (vectorX),
 *          for the input data there has to be a fixed value x0 = 1. This is
 *          because we have to estimate a vector of parameters [b0, b1, b2, ...]
 *          of n + 1 elements (because of the parameter that is not being
 *          multiplied by an x). n is the number of features of the vectorXs.
 *          So by defining x0 = 1 we have the same number of elements in the two
 *          vectors.
 */

var intelligentAgent = (function(){

    // Private members //

    var sampleElements = [], // [ [[x0,x1,x2,x3,...], y] ,[[x0,x1,x2,x3,...], y],... ]
        parameters = [];     // [b0, b1, b2, ..., bn]

    // f(vectorX) = b[0] * vectorX[0] + b[1] * vectorX[1] + ... + b[n] * vectorX[n]
    // where vectorX[0] = 1, this way the vector parameter and variable have the same length
    function predictionFunction (vectorX){
        var result = 0;
        for (var i = 0, numParameters = parameters.length; i < numParameters; ++i){
            result += parameters[i] * vectorX[i];
        }
        return result;
    }

    // This is the partial derivative of the cost function (we are taking the least
    // square error as our cost function)
    function rateOfChangeOfCost (variableChanging){
        var result = 0,
            numElementsInSample = sampleElements.length;

        for (var i = 0; i < numElementsInSample; ++i){
            result += ( predictionFunction(sampleElements[i][0]) - sampleElements[i][1] )
                    * sampleElements[i][0][variableChanging];
        }
        return (result * 2 / numElementsInSample);
    }

    // Gradient descent algorithm to find (an aproximation to)
    // the optimal values for the parameters of the linear function
    // by minimizing the cost function
    function findOptimalParameters (numIterations, learningRate){
        var numParameters = parameters.length,
            auxParameters = new Array(numParameters);
        // We need auxParameters because the parameters have to be
        // computed using the same values for each iteration. In case
        // we didn't use the auxParameters we would compute the first
        // parameter and then, for the rest we would use the rateOfChangeOfCost
        // with a different set of parameters than for the previous ones D:

        // Initialize auxParameters to the initial value of parameters
        for (var i = 0; i < numParameters; ++i){
            auxParameters[i] = parameters[i];
        }

        // Run actual gradient descent
        for (var i = 0; i < numIterations; ++i){
            // Compute the value for every parameter
            for (var j = 0; j < numParameters;++j){
                auxParameters[j] -= learningRate * rateOfChangeOfCost(j);
            }
            // Update the value of the parameters
            for (var j = 0; j < numParameters; ++j){
                parameters[j] = auxParameters[j];
            }
        }
    }

    // Public members //

    return ({
        train : function(sample, numIterations, learningRate){
            // Look for the first element X vector and see
            // how many parameters there are.
            var numParameters = sample[0][0].length;

            // Initialize the sample elements and the parameters
            sampleElements = sample;
            parameters = [];
            for (var i = 0; i < numParameters; ++i){
                parameters.push(0);
            }

            // Run gradient descent to find an aproximation to the
            // optimal parameters for the prediction function
            findOptimalParameters(numIterations, learningRate);

            console.log("Sample: ", sample);
            console.log("Parameters: ", parameters);
        },

        predict : function (vectorX){
            return predictionFunction(vectorX);
        }
    });

})();


var canvas = document.getElementById("canvas"),
    context = canvas.getContext("2d"),
    displayButton = document.getElementById("display-line-button"),
    clearButton = document.getElementById("clear-canvas-button"),
    sample = [];

// We want to map the points in the canvas to a domain [0,10]
// so we work with low values and they are in the same domain.
// PositionX = (canvasPositionX / canvasWidth) * 10
// For the y coordinate we have to substract the canvasHeight
// to have the origin at bottom
// PositionY = ((canvasHeight - canvasPositionY) / canvasHeight) * 10
var domainSize = 10,
    mapFactorX = canvas.width / domainSize,
    mapFactorY = canvas.height / domainSize;

context.fillStyle = "#000";

// Add elements to the sample and draw points in canvas when user clicks
canvas.addEventListener("click", function(event){
    var x = event.clientX / mapFactorX,
        y = (canvas.height - event.clientY ) / mapFactorY;
    sample.push( [[1, x], y] );
    context.fillRect( event.clientX, event.clientY, 5, 5 );
});

// Give the sample to the agent to train and display the function
displayButton.addEventListener("click", function(){
    if (sample.length > 0){
        intelligentAgent.train(sample, 100000, 0.01);

        var y0 = intelligentAgent.predict([1,0]);
        var y10 = intelligentAgent.predict([1,domainSize]);

        context.beginPath();
        context.moveTo(0, canvas.height - y0*mapFactorY);
        context.lineTo(domainSize*mapFactorX, canvas.height - y10*mapFactorY);
        context.stroke();
    }
});

// Clear sample and canvas
clearButton.addEventListener("click", function(){
   sample = [];
   context.clearRect(0,0,canvas.width, canvas.height);
});

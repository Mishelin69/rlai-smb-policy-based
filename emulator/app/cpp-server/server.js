const addon = require('../build/Release/testaddon.node');

console.log("Addon: ", addon.WrappedStack);
const test_obj = new addon.WrappedStack(10);

// server.js
const express = require('express');
const app = express();
const port = 3002;

app.use(express.json());

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', 'http://localhost:8080');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  next();
});

// app.use(express.static('public', { maxAge: 31557600000 }));

// app.get('/rlagent', (req, res) => {
//   try {
//     // Your logic to generate or fetch data goes here
//     const jsonData = { message: 'Hello from the data endpoint!' };
//     res.status(200).json(jsonData);
//   } catch (error) {
//     console.error('Error generating/fetching data:', error);
//     res.status(500).json({ error: 'Failed to fetch data' });
//   }
// });

app.post('/rlagent/predict', (req, res) => {
    try {
    // Access JSON input from the request body
    const inputData = req.body;
    // console.log('Received JSON input:', inputData);

    // Your logic to process the input and generate a response
    const responseData = { "keyCode": 67 };

    console.log("Current: ", test_obj.current());
    
    res.status(200).json(responseData);

  } catch (error) {
    console.error('Error processing data:', error);
    res.status(500).json({ error: 'Failed to process data' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
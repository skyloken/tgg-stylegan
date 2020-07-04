import { Box, CssBaseline } from '@material-ui/core';
import React from 'react';
import './App.css';
import Generate from './components/Generate';


function App() {
  return (
    <div className="App">
      <CssBaseline />
      <Box m={3}>
        <Generate />
      </Box>
    </div>
  );
}

export default App;

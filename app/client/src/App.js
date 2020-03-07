import React from 'react';
import { Grid, Box, Container, CssBaseline, Divider } from '@material-ui/core';
import './App.css';
import Generate from './components/Generate';
import Mix from './components/Mix';


function App() {
  return (
    <div className="App">
      <CssBaseline />
      <Box m={5}>
        <Grid container spacing={10} justify='center' alignItems='center'>
          <Grid item xs={6}>
            <Generate />
          </Grid>
          <Divider orientation="vertical" flexItem />
          <Grid item xs={5}>
            <Mix />
          </Grid>
        </Grid>
      </Box>
    </div>
  );
}

export default App;

import React from 'react';
import { Box, Container, CssBaseline, Divider } from '@material-ui/core';
import './App.css';
import GeneratePage from './components/GeneratePage';
import MixPage from './components/MixPage';


function App() {
  return (
    <div className="App">
      <CssBaseline />
      <Container fixed>
        <GeneratePage />
        <Divider />
        <MixPage />
      </Container>
    </div>
  );
}

export default App;

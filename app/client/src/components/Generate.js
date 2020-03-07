import React from "react";
import Image from "./Image";
import { Box, Grid, Button, Divider, Typography } from '@material-ui/core';
import _cloneDeep from 'lodash/cloneDeep';
import GeneratedImage from './GeneratedImage';
import GeneratedImageList from './GeneratedImageList';

class Generate extends React.Component {

    state = {
        japaneseGeneratedImages: [],
        westernGeneratedImages: []
    }

    componentWillMount() {

        ['japanese', 'western'].map(style => {
            const params = new URLSearchParams({
                'n': 6,
                'style': style
            });

            fetch(`/generate?${params}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(generatedImage => generatedImage.selected = false)
                    data[0].selected = true;
                    const newState = Array.from(this.state);
                    newState[`${style}GeneratedImages`] = data;
                    this.setState(newState);
                });
        });

    }

    handleJapaneseImageClick = (index) => {

        const newGeneratedImages = _cloneDeep(this.state.japaneseGeneratedImages);
        newGeneratedImages.forEach(generatedImage => generatedImage.selected = false);
        newGeneratedImages[index].selected = true;
        this.setState({
            japaneseGeneratedImages: newGeneratedImages
        });

    }

    handleWesternImageClick = (index) => {

        const newGeneratedImages = _cloneDeep(this.state.westernGeneratedImages);
        newGeneratedImages.forEach(generatedImage => generatedImage.selected = false);
        newGeneratedImages[index].selected = true;
        this.setState({
            westernGeneratedImages: newGeneratedImages
        });

    }

    render() {
        // console.log(this.state);

        return (
            <>
                <Typography variant="h5" align='center' gutterBottom>Japanese styles</Typography>
                <GeneratedImageList generatedImages={this.state.japaneseGeneratedImages} handleImageClick={this.handleJapaneseImageClick} />
                <Box m={3} />
                <Typography variant="h5" gutterBottom>Western styles</Typography>
                <GeneratedImageList generatedImages={this.state.westernGeneratedImages} handleImageClick={this.handleWesternImageClick} />
            </>
        );
    }

}

export default Generate;
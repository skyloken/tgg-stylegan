import { Box, CircularProgress, Divider, Grid, IconButton, Typography } from '@material-ui/core';
import RefreshIcon from '@material-ui/icons/Refresh';
import _cloneDeep from 'lodash/cloneDeep';
import React from "react";
import GeneratedImageList from './GeneratedImageList';
import Mix from './Mix';

class Generate extends React.Component {

    state = {
        japaneseGeneratedImages: [],
        westernGeneratedImages: []
    }

    fetchGeneratedImages(style) {
        const params = new URLSearchParams({
            'n': 6,
            'style': style
        });

        fetch(`/generate?${params}`)
            .then(response => response.json())
            .then(data => {
                data.forEach(image => image.selected = false)
                data[0].selected = true;
                const newState = _cloneDeep(this.state);
                newState[`${style}GeneratedImages`] = data;
                this.setState(newState);
            });
    }

    componentDidMount() {
        ['japanese', 'western'].forEach(style => this.fetchGeneratedImages(style));
    }

    handleJapaneseImageClick = (index) => {

        const newGeneratedImages = _cloneDeep(this.state.japaneseGeneratedImages);
        newGeneratedImages.forEach(image => image.selected = false);
        newGeneratedImages[index].selected = true;
        this.setState({
            japaneseGeneratedImages: newGeneratedImages
        });

    }

    handleWesternImageClick = (index) => {

        const newGeneratedImages = _cloneDeep(this.state.westernGeneratedImages);
        newGeneratedImages.forEach(image => image.selected = false);
        newGeneratedImages[index].selected = true;
        this.setState({
            westernGeneratedImages: newGeneratedImages
        });

    }

    handleRefreshButtonClick = (style) => {
        this.fetchGeneratedImages(style);
    }

    render() {

        return (
            <>
                <Grid container spacing={10} justify='center' alignItems='center'>
                    <Grid item xs={6}>
                        <Typography variant="h5" display='inline' gutterBottom>Japanese styles</Typography>
                        <IconButton onClick={() => this.handleRefreshButtonClick('japanese')} ><RefreshIcon /></IconButton>
                        {this.state.japaneseGeneratedImages.length === 0 ? <CircularProgress /> :
                            <GeneratedImageList
                                generatedImages={this.state.japaneseGeneratedImages}
                                handleImageClick={this.handleJapaneseImageClick}
                            />}
                        <Box m={3} />
                        <Typography variant="h5" display='inline' gutterBottom>Western styles</Typography>
                        <IconButton onClick={() => this.handleRefreshButtonClick('western')}><RefreshIcon /></IconButton>
                        {this.state.japaneseGeneratedImages.length === 0 ? <CircularProgress /> :
                            <GeneratedImageList
                                generatedImages={this.state.westernGeneratedImages}
                                handleImageClick={this.handleWesternImageClick}
                            />}
                    </Grid>
                    <Divider orientation="vertical" flexItem />
                    <Grid item xs={5}>
                        {(this.state.japaneseGeneratedImages.length === 0
                            || this.state.westernGeneratedImages.length === 0) ? <CircularProgress /> :
                            <Mix
                                jpnLatent={this.state.japaneseGeneratedImages.find(image => image.selected === true).latent}
                                wstLatent={this.state.westernGeneratedImages.find(image => image.selected === true).latent}
                            />}
                    </Grid>
                </Grid>
            </>
        );
    }

}

export default Generate;
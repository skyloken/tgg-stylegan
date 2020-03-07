import React from "react";
import Image from "./Image";
import { Box, Grid, GridList, Button, Divider } from '@material-ui/core';
import GeneratedImage from './GeneratedImage';

class GeneratePage extends React.Component {

    state = {
        genNum: 16,
        generatedImages: [],
        selectedImages: []
    }

    componentWillMount() {
        const params = new URLSearchParams({
            'n': this.state.genNum
        });

        fetch(`/generate?${params}`)
            .then(response => response.json())
            .then(data => this.setState({
                generatedImages: data
            }));
    }

    addSelectedImage = (selectedImage) => {
        if (this.state.selectedImages.length < 2) {
            this.setState({
                selectedImages: [...this.state.selectedImages, selectedImage]
            })
        }
    }

    render() {

        return (
            <>
                <Grid container spacing={5} justify='center'>
                    {
                        this.state.generatedImages.map((generatedImage, i) =>
                            <Grid item>
                                <GeneratedImage image={generatedImage} key={i} addSelectedImage={this.addSelectedImage} />
                            </Grid>
                        )}
                </Grid>
                <Divider />
                {
                    this.state.selectedImages.map((selectedImage, i) =>
                        <Grid item>
                            <Image base64={selectedImage.base64} key={i} />
                        </Grid>
                    )}
                <Button variant="contained" color="primary">Mix</Button>
                <Button variant="contained" color="primary">Refresh</Button>
            </>
        );
    }

}

export default GeneratePage;
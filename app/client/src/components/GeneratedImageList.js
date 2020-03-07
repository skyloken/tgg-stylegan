import React from 'react';
import { Grid } from '@material-ui/core';
import GeneratedImage from './GeneratedImage';

const GeneratedImageList = (props) => {

    return (
        <Grid container spacing={2} justify='center'>
            {props.generatedImages.map((generatedImage, i) =>
                <Grid item xs={4}>
                    <GeneratedImage image={generatedImage} key={i} onClick={() => props.handleImageClick(i)} />
                </Grid>
            )}
        </Grid>
    );

}

export default GeneratedImageList;

import React from 'react';
import { Typography, Slider, Box } from '@material-ui/core';
import Image from './Image';


class Mix extends React.Component {

    state = {
        mixedImages: [],
        mix_index: 15
    }

    componentWillMount() {
        const params = new URLSearchParams({
            'latent1': 'hoge',
            'latent2': 'fuga'
        });

        fetch(`/mix?${params}`)
            .then(response => response.json())
            .then(data => this.setState({
                mixedImages: data
            }));
    }

    handleSliderChange = (event, newValue) => {
        this.setState({
            mix_index: newValue
        });
    };

    render() {
        console.log(this.state);
        return (
            <>
                <Typography variant="h5" align='center' gutterBottom>Mixed style</Typography>
                <Image base64={this.state.mixedImages[this.state.mix_index]} width='100%' />
                <Box m={5} />
                <Typography>Mixing ratio</Typography>
                <Slider
                    value={this.state.mix_index}
                    aria-labelledby="discrete-slider"
                    step={1}
                    marks
                    min={0}
                    max={29}
                    onChange={this.handleSliderChange}
                />
            </>

        );


    }


}

export default Mix;
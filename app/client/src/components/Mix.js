
import React from 'react';
import { CircularProgress, Typography, Slider, Box } from '@material-ui/core';
import Image from './Image';


class Mix extends React.Component {

    state = {
        mixedImages: [],
        mix_index: 4
    }

    fetchMixedImages() {
        const body = {
            'wstLatent': this.props.wstLatent,
            'jpnLatent': this.props.jpnLatent
        };

        fetch('/mix', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json; charset=utf-8',
            },
            body: JSON.stringify(body)
        })
            .then(response => response.json())
            .then(data => this.setState({
                mixedImages: data
            }));
    }

    componentDidMount() {
        this.fetchMixedImages();
    }

    componentDidUpdate(prevProps) {
        if (this.props.jpnLatent !== prevProps.jpnLatent || this.props.wstLatent !== prevProps.wstLatent) {
            this.setState({
                mixedImages: []
            });
            this.fetchMixedImages();
        }
    }

    handleSliderChange = (event, newValue) => {
        this.setState({
            mix_index: newValue
        });
    };

    render() {

        return (
            <>
                {this.state.mixedImages.length === 0 ? <CircularProgress /> :
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
                    </>}
            </>
        );

    }


}

export default Mix;

import React from 'react';
import { Slider } from '@material-ui/core';
import GeneratedImage from './GeneratedImage';
import Image from './Image';


class MixPage extends React.Component {

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
                <Image base64={this.state.mixedImages[this.state.mix_index]} />
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

export default MixPage;
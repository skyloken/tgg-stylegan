import React, { Component } from 'react';

const API = 'http://localhost:5000/hello';

class Image extends Component {

    state = {
        base64: '',
    };

    componentDidMount() {
        fetch(API, {mode: 'cors'})
            .then(response => response.json())
            .then(data => this.setState({ base64: data.imageBase64 }));
    }

    render() {
        return <>
            <img src={'data:image/png;base64,' + this.state.base64} alt=''></img>
        </>;
    }


}
export default Image;
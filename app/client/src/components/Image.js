import React from 'react';
import './Image.css';

const Image = (props) => {
    return <img
        className={props.selected ? 'Image-selected' : 'Image'}
        src={'data:image/png;base64,' + props.base64}
        alt=''
        onClick={props.onClick}
        width={props.width}
        height={props.height} />
}

export default Image;
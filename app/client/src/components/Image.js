import React from 'react';
import './Image.css';

function Image(props) {
    return <img className='Image' src={'data:image/png;base64,' + props.base64} alt='' onClick={props.onClick} width={props.width} height={props.height} />
}

export default Image;
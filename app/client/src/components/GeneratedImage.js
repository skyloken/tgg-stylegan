import React from 'react';

import Tilt from 'react-tilt'
import Image from './Image';

const GeneratedImage = (props) => {

    return <Tilt className="Tilt" options={{ max: 25 }}>
        <div className="Tilt-inner">
            <Image selected={props.image.selected} base64={props.image.base64} width='100%' onClick={props.onClick} />
        </div>
    </Tilt>
    
}

export default GeneratedImage;
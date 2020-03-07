import React from 'react';

import Tilt from 'react-tilt'
import Image from './Image';

function GeneratedImage(props) {

    const handleClick = () => {
        props.addSelectedImage(props.image)
    }

    return <Tilt className="Tilt" options={{ max: 25 }}>
        <div className="Tilt-inner">
            <Image base64={props.image.base64} width={128} height={128} onClick={handleClick} />
        </div>
    </Tilt>
}
export default GeneratedImage;
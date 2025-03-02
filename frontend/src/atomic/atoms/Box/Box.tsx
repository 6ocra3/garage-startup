import React from 'react'
import styles from './Box.module.scss'
interface BoxProps {
    backgroundColor?: string
    title?: string
    comments?: string[]
    hint?: string
}
const Box: React.FC<BoxProps> = ({
    title,
    backgroundColor,
    comments,
    hint,
}) => {
    return (
        <div className={styles.box} style={{ backgroundColor }}>
            {hint && <div className={styles.hint}>{hint}</div>}
            <div className={styles.title}>{title}</div>
            <div className={styles.comments}>
                {comments?.map((item, index) => (
                    <div className={styles.comment} key={index}>
                        {item}
                    </div>
                ))}
            </div>
        </div>
    )
}

export default Box

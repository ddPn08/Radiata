import { ActionIcon, Box, Image } from '@mantine/core'
import { IconDownload } from '@tabler/icons-react'
import { ImageInformation } from 'internal:api'

import { downloadB64 } from '~/utils/download'

interface Props {
  image: string
  info: ImageInformation
  onClick: () => void
}

const GalleryImage = ({ image, info, onClick }: Props) => {
  return (
    <Box key={image} pos={'relative'}>
      <Image
        src={`data:image/png;base64,${image}`}
        alt={info.prompt}
        sx={{
          cursor: 'pointer',
        }}
        onClick={() => {
          onClick()
        }}
      />

      <ActionIcon
        variant={'light'}
        color={'gray'}
        pos={'absolute'}
        m={'xs'}
        right={0}
        bottom={0}
        onClick={() => {
          downloadB64(image, `${info.seed}.png`, 'image/png')
        }}
      >
        <IconDownload size={16} />
      </ActionIcon>
    </Box>
  )
}

export default GalleryImage

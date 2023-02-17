import { ActionIcon, Box, Image } from '@mantine/core'
import { IconDownload } from '@tabler/icons-react'
import type { ImageInformation } from 'internal:api'

import { downloadB64 } from '~/utils/download'

interface Props {
  image: string
  info: ImageInformation
  onClick: () => void
}

const imageUrl = (url: string) => {
  const regex = /^[A-Za-z0-9+/=]+$/
  return regex.test(url) ? `data:image/png;base64,${url}` : url
}

const GalleryImage = ({ image, info, onClick }: Props) => {
  return (
    <Box key={image} pos={'relative'}>
      <Image
        src={imageUrl(image)}
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

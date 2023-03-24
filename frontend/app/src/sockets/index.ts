import { io } from 'socket.io-client'

export const sockets = {
    diffusion: io('/diffusion'),
}
export const socket = io()
